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
#include "celeritas/GlobalGeoTestBase.hh"
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
using Action    = celeritas::MscInteraction::Action;

using celeritas::MemSpace;
using celeritas::Ownership;
using GeoParamsCRefDevice = celeritas::DeviceCRef<GeoParamsData>;
using GeoStateRefDevice   = celeritas::DeviceRef<GeoStateData>;

using SimStateValue = ::celeritas::HostVal<SimStateData>;
using SimStateRef   = ::celeritas::NativeRef<SimStateData>;

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
    using PhysicsParamsHostRef = ::celeritas::HostCRef<PhysicsParamsData>;
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::host>;

  protected:
    const char* geometry_basename() const override
    {
        return "four-steel-slabs";
    }

    void SetUp() override
    {
        RootImporter import_from_root(
            this->test_data_path("celeritas", "four-steel-slabs.root").c_str());
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
        EIonizationProcess::Options ioni_options;
        ioni_options.use_integral_xs = true;
        input.processes.push_back(std::make_shared<EIonizationProcess>(
            this->particle(), processes_data_, ioni_options));
        input.processes.push_back(std::make_shared<MultipleScatteringProcess>(
            this->particle(), this->material(), processes_data_));

        // Add action manager
        input.action_manager = this->action_mgr().get();

        return std::make_shared<PhysicsParams>(std::move(input));
    }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }

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

    // Test the step limitation algorithm and the msc sample scattering
    MscStep        step_result;
    MscInteraction sample_result;

    // Input
    static const real_type energy[] = {51.0231,
                                       10.0564,
                                       5.05808,
                                       1.01162,
                                       0.501328,
                                       0.102364,
                                       0.0465336,
                                       0.00708839};

    static const real_type step[] = {0.00279169,
                                     0.412343,
                                     0.0376414,
                                     0.078163296576415602,
                                     0.031624394625545782,
                                     0.002779271697902872,
                                     0.00074215289000934838,
                                     0.000031163160031423049};

    constexpr unsigned int nsamples = std::end(step) - std::begin(step);
    static_assert(nsamples == std::end(energy) - std::begin(energy),
                  "Input sizes do not match");

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
                               0,
                               TrackStatus::alive,
                               StepLimit{}};
        sim_state_data.push_back(state);
    }
    const SimStateRef& states = make_ref(states_ref);
    SimTrackView       sim_track_view(states, ThreadId{0});

    EXPECT_EQ(nsamples, sim_state_data.size());
    EXPECT_EQ(0, sim_track_view.num_steps());

    RandomEngine&       rng_engine = this->rng();
    std::vector<double> fstep;
    std::vector<double> angle;
    std::vector<double> displace;
    std::vector<char>   action;
    Real3               direction{0, 0, 1};

    for (auto i : celeritas::range(nsamples))
    {
        real_type r = i * 2 - real_type(1e-4);
        geo_view    = {{r, r, r}, direction};

        this->set_inc_particle(pdg::electron(), MevEnergy{energy[i]});

        UrbanMscStepLimit step_limiter(model->host_ref(),
                                       *part_view_,
                                       phys,
                                       material_view.material_id(),
                                       sim_track_view.num_steps() == 0,
                                       geo_view.find_safety(),
                                       step[i]);

        step_result = step_limiter(rng_engine);

        UrbanMscScatter scatter(model->host_ref(),
                                *part_view_,
                                &geo_view,
                                phys,
                                material_view,
                                step_result,
                                /* geo_limited = */ false);

        sample_result = scatter(rng_engine);

        fstep.push_back(sample_result.step_length);
        angle.push_back(sample_result.action != Action::unchanged
                            ? sample_result.direction[0]
                            : 0);
        displace.push_back(sample_result.action == Action::displaced
                               ? sample_result.displacement[0]
                               : 0);
        action.push_back(sample_result.action == Action::displaced   ? 'd'
                         : sample_result.action == Action::scattered ? 's'
                                                                     : 'u');
    }

    static const double expected_fstep[] = {0.0027916899999997,
                                            0.134631648532277,
                                            0.0376414,
                                            0.0781632965764156,
                                            0.00137047789888519,
                                            9.65987190264274e-05,
                                            0.000742152890009348,
                                            3.1163160031423e-05};
    EXPECT_VEC_SOFT_EQ(expected_fstep, fstep);
    static const double expected_angle[] = {0.000314741326035635,
                                            0.738624667826603,
                                            -0.145610123961716,
                                            0,
                                            -0.326505138945708,
                                            0.0130719743269634,
                                            0,
                                            0};
    EXPECT_VEC_NEAR(expected_angle, angle, 1e-10);
    static const double expected_displace[] = {8.19862035797085e-06,
                                               9.7530617641316e-05,
                                               -7.1670542039709e-05,
                                               0,
                                               -9.11378237130022e-05,
                                               9.7877513962823e-06,
                                               0,
                                               0};
    EXPECT_VEC_NEAR(expected_displace, displace, 1e-10);
    static const char expected_action[]
        = {'d', 'd', 'd', 'u', 'd', 'd', 'u', 'u'};
    EXPECT_VEC_EQ(expected_action, action);
}
