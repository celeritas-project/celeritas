//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UrbanMsc.test.cc
//---------------------------------------------------------------------------//
#include <random>

#include "base/CollectionStateStore.hh"
#include "base/Range.hh"
#include "geometry/GeoData.hh"
#include "geometry/GeoParams.hh"
#include "geometry/GeoTestBase.hh"
#include "geometry/GeoTrackView.hh"
#include "io/ImportData.hh"
#include "io/RootImporter.hh"
#include "physics/base/ImportedProcessAdapter.hh"
#include "physics/base/Model.hh"
#include "physics/base/ParticleData.hh"
#include "physics/base/PhysicsParams.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/em/EIonizationProcess.hh"
#include "physics/em/MultipleScatteringProcess.hh"
#include "physics/em/UrbanMscModel.hh"
#include "physics/em/detail/UrbanMscScatter.hh"
#include "physics/em/detail/UrbanMscStepLimit.hh"
#include "physics/grid/RangeCalculator.hh"
#include "random/DiagnosticRngEngine.hh"

#include "celeritas_test.hh"
#include "gtest/Test.hh"

using namespace celeritas;
using namespace celeritas_test;

using VGT       = ValueGridType;
using MevEnergy = units::MevEnergy;
using detail::UrbanMscScatter;
using detail::UrbanMscStepLimit;

using celeritas::MemSpace;
using celeritas::Ownership;
using GeoParamsCRefDevice
    = celeritas::GeoParamsData<Ownership::const_reference, MemSpace::device>;
using GeoStateRefDevice
    = celeritas::GeoStateData<Ownership::reference, MemSpace::device>;
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UrbanMscTest : public GeoTestBase<celeritas::GeoParams>
{
  public:
    const char* dirname() const override { return "io"; }
    const char* filebase() const override { return "geant-exporter-data"; }

  protected:
    using RandomEngine = celeritas_test::DiagnosticRngEngine<std::mt19937>;

    using SPConstMaterials = std::shared_ptr<const MaterialParams>;
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using SPConstPhysics   = std::shared_ptr<const PhysicsParams>;
    using SPConstImported  = std::shared_ptr<const ImportedProcesses>;

    using PhysicsStateStore
        = CollectionStateStore<PhysicsStateData, MemSpace::host>;
    using ParticleStateStore
        = CollectionStateStore<ParticleStateData, MemSpace::host>;
    using PhysicsParamsHostRef
        = PhysicsParamsData<Ownership::const_reference, MemSpace::host>;
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::host>;

    void SetUp() override
    {
        RootImporter import_from_root(
            this->test_data_path("io", "geant-exporter-data.root").c_str());
        auto data = import_from_root();

        particle_params_ = ParticleParams::from_import(data);
        material_params_ = MaterialParams::from_import(data);
        processes_data_
            = std::make_shared<ImportedProcesses>(std::move(data.processes));

        CELER_ENSURE(particle_params_);
        CELER_ENSURE(processes_data_->size() > 0);

        PhysicsParams::Input input;
        input.particles = particle_params_;
        input.materials = material_params_;

        // Add EIonizationProcess and MultipleScatteringProcess
        input.processes.push_back(std::make_shared<EIonizationProcess>(
            particle_params_, processes_data_));
        input.processes.push_back(std::make_shared<MultipleScatteringProcess>(
            particle_params_, material_params_, processes_data_));

        physics_params_ = std::make_shared<PhysicsParams>(std::move(input));

        // Make one state per particle
        auto state_size = particle_params_->size();

        CELER_ASSERT(physics_params_);
        params_ref_     = physics_params_->host_ref();
        physics_state_  = PhysicsStateStore(*physics_params_, state_size);
        particle_state_ = ParticleStateStore(*particle_params_, state_size);
        geo_state_      = GeoStateStore(*this->geometry(), 1);
    }

    // Make physics track view
    PhysicsTrackView make_track_view(const char* particle, MaterialId mid)
    {
        CELER_EXPECT(particle && mid);

        auto pid = this->particle_params_->find(particle);
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
        CELER_EXPECT(particle_params_);
        CELER_EXPECT(pdg);
        CELER_EXPECT(energy >= zero_quantity());

        // Construct track view
        part_view_ = std::make_shared<ParticleTrackView>(
            particle_params_->host_ref(), particle_state_.ref(), ThreadId{0});

        // Initialize
        ParticleTrackView::Initializer_t init;
        init.particle_id = particle_params_->find(pdg);
        init.energy      = energy;
        *part_view_      = init;
    }

    RandomEngine& rng()
    {
        rng_.reset_count();
        return rng_;
    }

    SPConstMaterials material_params_;
    SPConstParticles particle_params_;
    SPConstPhysics   physics_params_;
    SPConstImported  processes_data_;

    PhysicsParamsHostRef params_ref_;
    PhysicsStateStore    physics_state_;
    ParticleStateStore   particle_state_;
    GeoStateStore        geo_state_;
    // Views
    std::shared_ptr<ParticleTrackView> part_view_;
    RandomEngine                       rng_;

    std::shared_ptr<UrbanMscModel> model_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UrbanMscTest, msc_scattering)
{
    // Views
    PhysicsTrackView   phys     = this->make_track_view("e-", MaterialId{1});
    GeoTrackView       geo_view = this->make_geo_track_view();
    const MaterialView material_view = material_params_->get(MaterialId{1});

    // Create the model
    std::shared_ptr<UrbanMscModel> model = std::make_shared<UrbanMscModel>(
        ModelId{0}, *particle_params_, *material_params_);

    // Check MscMaterialDara for the current material (G4_STAINLESS-STEEL)
    const detail::UrbanMscMaterialData& msc_
        = model->host_ref().msc_data[material_view.material_id()];

    EXPECT_DOUBLE_EQ(msc_.zeff, 25.8);
    EXPECT_DOUBLE_EQ(msc_.z23, 8.7313179636909233);
    EXPECT_DOUBLE_EQ(msc_.coeffth1, 0.97326969977637379);
    EXPECT_DOUBLE_EQ(msc_.coeffth2, 0.044188139325421663);
    EXPECT_DOUBLE_EQ(msc_.d[0], 1.6889578380303167);
    EXPECT_DOUBLE_EQ(msc_.d[1], 2.745018223507488);
    EXPECT_DOUBLE_EQ(msc_.d[2], -2.2531516772497562);
    EXPECT_DOUBLE_EQ(msc_.d[3], 0.052696806851297018);
    EXPECT_DOUBLE_EQ(msc_.stepmin_a, 4.4449610414595817);
    EXPECT_DOUBLE_EQ(msc_.stepmin_b, 1.5922149179564158);
    EXPECT_DOUBLE_EQ(msc_.d_over_r, 0.64474963087322135);
    EXPECT_DOUBLE_EQ(msc_.d_over_r_mh, 1.1248191999999999);

    // Test the step limitation algorithm and the msc sample scattering
    detail::MscStepLimiterResult step_result;
    detail::MscSamplingResult    sample_result;

    // Input
    const int nsamples = 8;

    real_type energy[nsamples] = {51.0231,
                                  10.0564,
                                  5.05808,
                                  1.01162,
                                  0.501328,
                                  0.102364,
                                  0.0465336,
                                  0.00708839};

    real_type step[nsamples] = {0.00279169,
                                0.412343,
                                0.0376414,
                                0.178529,
                                0.0836231,
                                0.125696,
                                0.00143809,
                                0.105187};

    RandomEngine&       rng_engine = this->rng();
    std::vector<double> fstep;
    std::vector<double> angle;
    Real3               direction{0, 0, 1};

    for (auto i : celeritas::range(nsamples))
    {
        real_type r = i * 2.0 - 1e-4;
        geo_view    = {{r, r, r}, direction};

        this->set_inc_particle(pdg::electron(), MevEnergy{energy[i]});
        phys.step_length(step[i]);

        UrbanMscStepLimit step_limiter(
            model->host_ref(), *part_view_, &geo_view, phys, material_view);

        step_result = step_limiter(rng_engine);

        UrbanMscScatter scatter(model->host_ref(),
                                *part_view_,
                                direction,
                                phys,
                                material_view,
                                step_result);

        sample_result = scatter(rng_engine);

        fstep.push_back(sample_result.step_length);
        angle.push_back(sample_result.direction[0]);
    }

    const double expected_fstep[] = {0.0027916899999997,
                                     0.0409943838810989,
                                     0.023815696237744,
                                     0.0357277834605262,
                                     0.000860509603114591,
                                     8.42039426529163e-05,
                                     0.000286789820693628,
                                     1.17373195131411e-05};

    const double expected_angle[] = {0.000314741326035635,
                                     0.394369575335092,
                                     -0.111678511182682,
                                     -0.657415795200799,
                                     0.103369072552411,
                                     -0.183547381906496,
                                     0.793645871128744,
                                     -0.98020130119347};

    EXPECT_VEC_SOFT_EQ(expected_fstep, fstep);
    EXPECT_VEC_SOFT_EQ(expected_angle, angle);
}
