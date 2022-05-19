//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/UrbanMsc.test.cc
//---------------------------------------------------------------------------//
#include <random>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/Ref.hh"
#include "celeritas/GlobalTestBase.hh"
#include "celeritas/em/distribution/UrbanMscScatter.hh"
#include "celeritas/em/distribution/UrbanMscStepLimit.hh"
#include "celeritas/em/model/UrbanMscModel.hh"
#include "celeritas/em/process/EIonizationProcess.hh"
#include "celeritas/em/process/MultipleScatteringProcess.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/ActionManager.hh"
#include "celeritas/grid/RangeCalculator.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Model.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/track/SimData.hh"
#include "celeritas/track/SimTrackView.hh"

#include "DiagnosticRngEngine.hh"
#include "Test.hh"
#include "celeritas_test.hh"

using namespace celeritas;

using VGT       = ValueGridType;
using MevEnergy = units::MevEnergy;

using celeritas::MemSpace;
using celeritas::Ownership;
using GeoParamsCRefDevice
    = celeritas::GeoParamsData<Ownership::const_reference, MemSpace::device>;
using GeoStateRefDevice
    = celeritas::GeoStateData<Ownership::reference, MemSpace::device>;

using SimStateValue = SimStateData<Ownership::value, MemSpace::host>;
using SimStateRef   = SimStateData<Ownership::reference, MemSpace::native>;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UrbanMscTest : public celeritas_test::GlobalTestBase
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
    const char* geometry_basename() const override
    {
        return "g4-ext-testem15";
    }

    void SetUp() override
    {
        RootImporter import_from_root(
            this->test_data_path("celeritas", "g4-ext-testem15.root").c_str());
        import_data_    = import_from_root();
        processes_data_ = std::make_shared<ImportedProcesses>(
            std::move(import_data_.processes));
        CELER_ASSERT(processes_data_->size() > 0);

        // Make one state per particle
        auto state_size = this->particle()->size();

        params_ref_     = this->physics()->host_ref();
        physics_state_  = PhysicsStateStore(*this->physics(), state_size);
        particle_state_ = ParticleStateStore(*this->particle(), state_size);
        geo_state_      = GeoStateStore(*this->geometry(), 1);
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
            params_ref_, physics_state_.ref(), pid, mid, tid);
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

    PhysicsParamsHostRef params_ref_;
    PhysicsStateStore    physics_state_;
    ParticleStateStore   particle_state_;
    GeoStateStore        geo_state_;

    // Views
    std::shared_ptr<ParticleTrackView> part_view_;
    RandomEngine                       rng_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UrbanMscTest, msc_scattering)
{
    // Views
    PhysicsTrackView   phys     = this->make_track_view("e-", MaterialId{1});
    GeoTrackView       geo_view = this->make_geo_track_view();
    const MaterialView material_view = this->material()->get(MaterialId{1});

    // Create the model
    std::shared_ptr<UrbanMscModel> model = std::make_shared<UrbanMscModel>(
        ActionId{0}, *this->particle(), *this->material());

    // Check MscMaterialDara for the current material (G4_STAINLESS-STEEL)
    const UrbanMscMaterialData& msc_
        = model->host_ref().msc_data[material_view.material_id()];

    EXPECT_DOUBLE_EQ(msc_.zeff, 25.8);
    EXPECT_DOUBLE_EQ(msc_.z23, 8.7313179636909233);
    EXPECT_DOUBLE_EQ(msc_.coeffth1, 0.97326969977637379);
    EXPECT_DOUBLE_EQ(msc_.coeffth2, 0.044188139325421663);
    EXPECT_DOUBLE_EQ(msc_.d[0], 1.6889578380303167);
    EXPECT_DOUBLE_EQ(msc_.d[1], 2.745018223507488);
    EXPECT_DOUBLE_EQ(msc_.d[2], -2.2531516772497562);
    EXPECT_DOUBLE_EQ(msc_.d[3], 0.052696806851297018);
    EXPECT_DOUBLE_EQ(msc_.stepmin_a, 1e3 * 4.4449610414595817);
    EXPECT_DOUBLE_EQ(msc_.stepmin_b, 1e3 * 1.5922149179564158);
    EXPECT_DOUBLE_EQ(msc_.d_over_r, 0.64474963087322135);
    EXPECT_DOUBLE_EQ(msc_.d_over_r_mh, 1.1248191999999999);

    // Test the step limitation algorithm and the msc sample scattering with
    // respect to TestEM15 with G4_STAINLESS-STEEL and 1mm cut: For details,
    // refer to Geant4 Release 11.0 examples/extended/electromagnetic/TestEm15

    // TestEM15 parameters
    constexpr unsigned int nsamples = 1e+5;
    constexpr double       det_size = 1e+5 * units::millimeter;
    Real3                  origin{-det_size / 2, 0, 0};
    Real3                  direction{1, 0, 0};

    // Mock SimStateData
    SimStateValue states_ref;
    auto          sim_state_data = make_builder(&states_ref.state);
    sim_state_data.reserve(nsamples);

    for (unsigned int i : celeritas::range(nsamples))
    {
        SimTrackState state = {TrackId{i},
                               TrackId{i},
                               EventId{1},
                               i % 2,
                               TrackStatus::alive,
                               StepLimit{}};
        sim_state_data.push_back(state);
    }
    const SimStateRef& states = make_ref(states_ref);
    SimTrackView       sim_track_view(states, ThreadId{0});

    EXPECT_EQ(nsamples, sim_state_data.size());
    EXPECT_EQ(0, sim_track_view.num_steps());

    // Input energy
    static const real_type energy[]   = {100, 10, 1, 1e-1, 1e-2, 1e-3};
    constexpr unsigned int num_energy = std::end(energy) - std::begin(energy);

    // Test variables
    std::vector<double> geom_path;
    std::vector<double> true_path;
    std::vector<double> lateral_dist;
    std::vector<double> psi_mean;
    std::vector<double> mom_xdir;
    std::vector<double> phi_correl;

    RandomEngine& rng_engine = this->rng();

    MscStep        step_result;
    MscInteraction sample_result;

    for (real_type e_inc : energy)
    {
        double sum_true_path    = 0;
        double sum_geom_path    = 0;
        double sum_lateral_dist = 0;
        double sum_psi          = 0;
        double sum_mom_xdir     = 0;
        double sum_phi_correl   = 0;

        for (CELER_MAYBE_UNUSED unsigned int j : celeritas::range(nsamples))
        {
            this->set_inc_particle(pdg::electron(), MevEnergy{e_inc});
            geo_view = {origin, direction};

            // Sample multiple scattering step limit
            UrbanMscStepLimit step_limiter(model->host_ref(),
                                           *part_view_,
                                           &geo_view,
                                           phys,
                                           material_view,
                                           sim_track_view.num_steps() == 0,
                                           det_size);

            step_result = step_limiter(rng_engine);

            // Mock transportation
            geo_view
                = {{-det_size / 2 + step_result.geom_path, 0, 0}, direction};

            // Sample the multiple scattering
            UrbanMscScatter scatter(model->host_ref(),
                                    *part_view_,
                                    &geo_view,
                                    phys,
                                    material_view,
                                    step_result);

            sample_result = scatter(rng_engine);

            // Geometrical path length
            sum_geom_path += step_result.geom_path;

            // True path length
            sum_true_path += sample_result.step_length;

            // Lateral displacement
            double disp_y      = sample_result.displacement[1];
            double disp_z      = sample_result.displacement[2];
            double lateral_arm = sqrt(ipow<2>(disp_y) + ipow<2>(disp_z));
            sum_lateral_dist += lateral_arm;

            // Psi variable
            sum_psi += std::atan(lateral_arm / step_result.geom_path);

            // Angle along the line of flight
            sum_mom_xdir += sample_result.direction[0];

            // Phi correlation
            if (lateral_arm > 0)
            {
                sum_phi_correl += (disp_y * sample_result.direction[1]
                                   + disp_z * sample_result.direction[2])
                                  / lateral_arm;
            }
        }

        // Mean values of test variables
        geom_path.push_back(sum_geom_path / nsamples);
        true_path.push_back(sum_true_path / nsamples);
        lateral_dist.push_back(sum_lateral_dist / nsamples);
        psi_mean.push_back(sum_psi / nsamples);
        mom_xdir.push_back(sum_mom_xdir / nsamples);
        phi_correl.push_back(sum_phi_correl / nsamples);
    }

    // Expected results obtained from TestEM15
    static const double g4_geom_path[]
        = {7.9736, 1.3991e-1, 2.8978e-3, 9.8068e-5, 1.9926e-6, 1.7734e-7};

    static const double g4_true_path[]
        = {8.8845, 1.5101e-1, 3.082e-3, 1.0651e-4, 2.1776e-6, 2.5102e-7};

    static const double g4_lateral_dist[]
        = {0, 4.1431e-2, 7.6514e-4, 3.0315e-5, 6.4119e-7, 1.2969e-7};

    static const double g4_psi_mean[]
        = {0, 0.28637, 0.25691, 0.29862, 0.31131, 0.63142};

    static const double g4_mom_xdir[]
        = {1, 0.83511, 0.86961, 0.84, 0.83257, 0.46774};

    static const double g4_phi_correl[]
        = {0, 0.37091, 0.32647, 0.35153, 0.34172, 0.58865};

    // Tolerance of the relative error with respect to Geant4: percent
    constexpr double tolerance = 0.01;
    for (auto i : celeritas::range(num_energy))
    {
        EXPECT_SOFT_NEAR(g4_geom_path[i], geom_path[i], tolerance);
        EXPECT_SOFT_NEAR(g4_true_path[i], true_path[i], tolerance);
        EXPECT_SOFT_NEAR(g4_lateral_dist[i], lateral_dist[i], tolerance);
        EXPECT_SOFT_NEAR(g4_psi_mean[i], psi_mean[i], tolerance);
        EXPECT_SOFT_NEAR(g4_mom_xdir[i], mom_xdir[i], tolerance);
        EXPECT_SOFT_NEAR(g4_phi_correl[i], phi_correl[i], tolerance);
    }
}
