//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/UrbanMsc.test.cc
//---------------------------------------------------------------------------//
#include "UrbanMsc.test.hh"

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/GlobalGeoTestBase.hh"
#include "celeritas/em/distribution/UrbanMscHelper.hh"
#include "celeritas/em/distribution/UrbanMscScatter.hh"
#include "celeritas/em/distribution/UrbanMscStepLimit.hh"
#include "celeritas/em/model/UrbanMscModel.hh"
#include "celeritas/em/process/EIonizationProcess.hh"
#include "celeritas/em/process/MultipleScatteringProcess.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/field/LinearPropagator.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/grid/RangeCalculator.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/random/RngParams.hh"

#include "DiagnosticRngEngine.hh"
#include "Test.hh"
#include "celeritas_test.hh"

using namespace celeritas;
using MevEnergy = units::MevEnergy;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UrbanMscTest : public celeritas_test::GlobalGeoTestBase
{
  protected:
    using RandomEngine    = celeritas_test::DiagnosticRngEngine<std::mt19937>;
    using SPConstImported = std::shared_ptr<const ImportedProcesses>;

    using PhysicsStateStore
        = CollectionStateStore<PhysicsStateData, MemSpace::host>;
    using ParticleStateStore
        = CollectionStateStore<ParticleStateData, MemSpace::host>;
    using PhysicsParamsHostRef
        = PhysicsParamsData<Ownership::const_reference, MemSpace::host>;
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::host>;

  protected:
    const char* geometry_basename() const { return "geant4-testem15"; }

    void SetUp() override
    {
        RootImporter import_from_root(
            this->test_data_path("celeritas", "geant4-testem15.root").c_str());
        import_data_    = import_from_root();
        processes_data_ = std::make_shared<ImportedProcesses>(
            std::move(import_data_.processes));
        CELER_ASSERT(processes_data_->size() > 0);

        // Make one state per particle
        auto state_size = this->particle()->size();

        geo_state_      = GeoStateStore(*this->geometry(), 1);
        physics_state_  = PhysicsStateStore(*this->physics(), state_size);
        particle_state_ = ParticleStateStore(*this->particle(), state_size);

        phys_params_ = this->physics()->host_ref();
        rng_params_  = std::make_shared<RngParams>(12345);

        // Test parameters with the TestEM15 detector (100 meter box)
        test_param.nstates   = 1e+5;
        test_param.position  = {-1e+5 * units::millimeter / 2 + 1e-8, 0, 0};
        test_param.direction = {1, 0, 0};

        // Create the Urban msc model
        model_ = std::make_shared<UrbanMscModel>(
            ActionId{0}, *this->particle(), *this->material());
    }

    SPConstParticle build_particle() override
    {
        return ParticleParams::from_import(import_data_);
    }

    SPConstMaterial build_material() override
    {
        return MaterialParams::from_import(import_data_);
    }

    SPConstPhysics build_physics() override
    {
        PhysicsParams::Input input;
        input.particles = this->particle();
        input.materials = this->material();

        // Add EIonizationProcess and MultipleScatteringProcess
        input.processes.push_back(std::make_shared<EIonizationProcess>(
            this->particle(), processes_data_));
        input.processes.push_back(std::make_shared<MultipleScatteringProcess>(
            this->particle(), this->material(), processes_data_));

        // Add action manager
        input.action_manager = this->action_mgr().get();

        return std::make_shared<PhysicsParams>(std::move(input));
    }

    SPConstGeoMaterial build_geomaterial() override
    {
        CELER_ASSERT_UNREACHABLE();
    }

    SPConstCutoff build_cutoff() override { CELER_ASSERT_UNREACHABLE(); }

    // Make physics track view
    PhysicsTrackView make_track_view(const char* particle, MaterialId mid)
    {
        CELER_EXPECT(particle && mid);

        auto pid = this->particle()->find(particle);
        CELER_ASSERT(pid);
        CELER_ASSERT(pid.get() < physics_state_.size());

        ThreadId tid((pid.get() + 1) % physics_state_.size());

        // Construct and initialize
        PhysicsTrackView phys_view(
            phys_params_, physics_state_.ref(), pid, mid, tid);
        phys_view = PhysicsTrackInitializer{};
        return phys_view;
    }

    //! Make geometry track view
    GeoTrackView make_geo_track_view()
    {
        return GeoTrackView(
            this->geometry()->host_ref(), geo_state_.ref(), ThreadId(0));
    }

    void set_inc_particle(PDGNumber pdg, MevEnergy energy)
    {
        CELER_EXPECT(this->particle());
        CELER_EXPECT(pdg);
        CELER_EXPECT(energy >= zero_quantity());

        // Construct track view
        part_view_ = std::make_shared<ParticleTrackView>(
            this->particle()->host_ref(), particle_state_.ref(), ThreadId{0});

        // Initialize
        ParticleTrackView::Initializer_t init;
        init.particle_id = this->particle()->find(pdg);
        init.energy      = energy;
        *part_view_      = init;
    }

    RandomEngine& rng()
    {
        rng_.reset_count();
        return rng_;
    }

    ImportData      import_data_;
    SPConstImported processes_data_;

    GeoStateStore        geo_state_;
    PhysicsStateStore    physics_state_;
    ParticleStateStore   particle_state_;
    PhysicsParamsHostRef phys_params_;

    // Views
    std::shared_ptr<ParticleTrackView> part_view_;

    // Random number generator
    std::shared_ptr<RngParams> rng_params_;
    RandomEngine               rng_;

    // Test parameters
    celeritas_test::MscTestParams test_param;

    // Msc model
    std::shared_ptr<UrbanMscModel> model_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(UrbanMscTest, msc_data)
{
    // Views
    const MaterialView material_view = this->material()->get(MaterialId{1});

    // Check MscMaterialDara for the current material (G4_STAINLESS-STEEL)
    const UrbanMscMaterialData& msc_data
        = model_->host_ref().msc_data[material_view.material_id()];

    EXPECT_DOUBLE_EQ(msc_data.zeff, 25.8);
    EXPECT_DOUBLE_EQ(msc_data.z23, 8.7313179636909233);
    EXPECT_DOUBLE_EQ(msc_data.coeffth1, 0.97326969977637379);
    EXPECT_DOUBLE_EQ(msc_data.coeffth2, 0.044188139325421663);
    EXPECT_DOUBLE_EQ(msc_data.d[0], 1.6889578380303167);
    EXPECT_DOUBLE_EQ(msc_data.d[1], 2.745018223507488);
    EXPECT_DOUBLE_EQ(msc_data.d[2], -2.2531516772497562);
    EXPECT_DOUBLE_EQ(msc_data.d[3], 0.052696806851297018);
    EXPECT_DOUBLE_EQ(msc_data.stepmin_a, 1e3 * 4.4449610414595817);
    EXPECT_DOUBLE_EQ(msc_data.stepmin_b, 1e3 * 1.5922149179564158);
    EXPECT_DOUBLE_EQ(msc_data.d_over_r, 0.64474963087322135);
    EXPECT_DOUBLE_EQ(msc_data.d_over_r_mh, 1.1248191999999999);
}

TEST_F(UrbanMscTest, msc_scattering)
{
    // Views
    PhysicsTrackView   phys     = this->make_track_view("e-", MaterialId{1});
    GeoTrackView       geo_view = this->make_geo_track_view();
    const MaterialView material_view = this->material()->get(MaterialId{1});

    // Test the step limitation algorithm and the msc sample scattering with
    // respect to TestEM15 using G4_STAINLESS-STEEL and 1mm cut: For details,
    // refer to Geant4 Release 11.0 examples/extended/electromagnetic/TestEm15

    // TestEM15 parameters
    unsigned int nsamples = this->test_param.nstates;

    // Input energy
    constexpr unsigned int num_energy = std::end(celeritas_test::energy)
                                        - std::begin(celeritas_test::energy);

    // Test output
    std::vector<celeritas_test::MscTestOutput> output;
    output.resize(nsamples);

    std::vector<celeritas_test::MscTestOutput> mean;
    mean.resize(num_energy);

    RandomEngine& rng_engine = this->rng();

    MscStep        step_result;
    MscInteraction sample_result;

    for (unsigned int j : celeritas::range(num_energy))
    {
        for (unsigned int i : celeritas::range(nsamples))
        {
            this->set_inc_particle(pdg::electron(),
                                   MevEnergy{celeritas_test::energy[j]});
            geo_view = {test_param.position, test_param.direction};

            UrbanMscHelper msc_helper(model_->host_ref(), *part_view_, phys);

            // Sample multiple scattering step limit
            UrbanMscStepLimit step_limiter(model_->host_ref(),
                                           *part_view_,
                                           phys,
                                           material_view.material_id(),
                                           true,
                                           geo_view.find_safety(),
                                           msc_helper.range());

            step_result = step_limiter(rng_engine);

            // Propagate up to the geometric step length
            real_type        geo_step = step_result.geom_path;
            LinearPropagator propagate(&geo_view);
            auto             propagated = propagate(geo_step);

            if (propagated.boundary)
            {
                // Stopped at a geometry boundary:
                step_result.geom_path = propagated.distance;
            }

            // Sample the multiple scattering
            UrbanMscScatter scatter(model_->host_ref(),
                                    *part_view_,
                                    &geo_view,
                                    phys,
                                    material_view,
                                    step_result);

            sample_result = scatter(rng_engine);

            output[i] = celeritas_test::calc_output(step_result, sample_result);
        }

        // Calculate the mean value of test variables
        mean[j] = celeritas_test::calc_mean(output);
    }

    // Verify results with respect to Geant4 TestEM15
    celeritas_test::check_result(mean);
}

//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//
#define UrbanMscTestDeviceTest TEST_IF_CELER_DEVICE(UrbanMscTestDeviceTest)
class UrbanMscDeviceTest : public UrbanMscTest
{
  public:
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::device>;
    using MaterialStateStore
        = CollectionStateStore<MaterialStateData, MemSpace::device>;
    using ParticleStateStore
        = CollectionStateStore<ParticleStateData, MemSpace::device>;
    using PhysicsStateStore
        = CollectionStateStore<PhysicsStateData, MemSpace::device>;
    using RngDeviceStore = CollectionStateStore<RngStateData, MemSpace::device>;
};

TEST_F(UrbanMscDeviceTest, TEST_IF_CELER_DEVICE(device_msc_scattering))
{
    // Input energy
    constexpr unsigned int num_energy = std::end(celeritas_test::energy)
                                        - std::begin(celeritas_test::energy);

    // Test variables
    std::vector<celeritas_test::MscTestOutput> mean;
    mean.resize(num_energy);

    // Setup test input
    celeritas_test::MscTestInput input;
    input.test_param = this->test_param;

    // Params
    input.geometry_params = this->geometry()->device_ref();
    input.material_params = this->material()->device_ref();
    input.particle_params = this->particle()->device_ref();
    input.physics_params  = this->physics()->device_ref();

    // States
    GeoStateStore      geo_states(*this->geometry(), test_param.nstates);
    ParticleStateStore part_states(*this->particle(), test_param.nstates);
    PhysicsStateStore  phys_states(*this->physics(), test_param.nstates);

    input.geometry_states = geo_states.ref();
    input.particle_states = part_states.ref();
    input.physics_states  = phys_states.ref();

    // Other input data
    RngDeviceStore rng_states(*rng_params_, test_param.nstates);
    input.rng_states = rng_states.ref();

    input.msc_data   = this->model_->device_ref();
    input.test_param = this->test_param;

    ParticleId pid = this->particle()->find("e-");

    for (CELER_MAYBE_UNUSED int i : celeritas::range(test_param.nstates))
    {
        input.init_phys.push_back({MaterialId{1}, pid});
    }

    for (unsigned int j : celeritas::range(num_energy))
    {
        for (CELER_MAYBE_UNUSED int i : celeritas::range(test_param.nstates))
        {
            input.init_part.push_back(
                {pid, MevEnergy{celeritas_test::energy[j]}});
        }
        // Run the Msc test kernel
        auto output = celeritas_test::msc_test(input);

        // Calculate the mean value of test variables
        mean[j] = celeritas_test::calc_mean(output);

        input.init_part.clear();
    }

    // Verify results with respect to Geant4 TestEM15
    celeritas_test::check_result(mean);
}
