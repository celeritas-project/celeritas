//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/UrbanMsc.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/msc/UrbanMsc.hh"

#include <random>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/RootTestBase.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/em/msc/UrbanMscScatter.hh"
#include "celeritas/em/msc/UrbanMscStepLimit.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/grid/RangeCalculator.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/track/SimData.hh"
#include "celeritas/track/SimTrackView.hh"

#include "DiagnosticRngEngine.hh"
#include "Test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using VGT = ValueGridType;
using MevEnergy = units::MevEnergy;
using Action = MscInteraction::Action;

using GeoParamsCRefDevice = DeviceCRef<GeoParamsData>;
using GeoStateRefDevice = DeviceRef<GeoStateData>;

using SimStateValue = HostVal<SimStateData>;
using SimStateRef = NativeRef<SimStateData>;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UrbanMscTest : public RootTestBase
{
  protected:
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;

    using PhysicsStateStore
        = CollectionStateStore<PhysicsStateData, MemSpace::host>;
    using ParticleStateStore
        = CollectionStateStore<ParticleStateData, MemSpace::host>;
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::host>;
    using SimStateStore = CollectionStateStore<SimStateData, MemSpace::host>;

  protected:
    char const* geometry_basename() const final { return "four-steel-slabs"; }

    void SetUp() override
    {
        // Load MSC data
        msc_params_ = UrbanMscParams::from_import(
            *this->particle(), *this->material(), this->imported_data());
        ASSERT_TRUE(msc_params_);

        // Allocate particle state
        auto state_size = 1;
        physics_state_
            = PhysicsStateStore(this->physics()->host_ref(), state_size);
        particle_state_
            = ParticleStateStore(this->particle()->host_ref(), state_size);
        geo_state_ = GeoStateStore(this->geometry()->host_ref(), state_size);
        sim_state_ = SimStateStore(state_size);
    }

    SPConstTrackInit build_init() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }

    // Initialize a track
    PhysicsTrackView
    make_track_view(PDGNumber pdg, MaterialId mid, MevEnergy energy)
    {
        CELER_EXPECT(pdg);
        CELER_EXPECT(mid);
        CELER_EXPECT(energy > zero_quantity());

        auto pid = this->particle()->find(pdg);
        CELER_ASSERT(pid);
        const ThreadId tid{0};

        // Initialize particle
        {
            ParticleTrackView par{
                this->particle()->host_ref(), particle_state_.ref(), tid};
            ParticleTrackView::Initializer_t init;
            init.particle_id = pid;
            init.energy = energy;
            par = init;
        }

        // Initialize physics
        PhysicsTrackView phys_view(
            this->physics()->host_ref(), physics_state_.ref(), pid, mid, tid);
        phys_view = PhysicsTrackInitializer{};

        // Calculate and store the energy loss (dedx) range limit
        auto ppid = phys_view.eloss_ppid();
        auto grid_id = phys_view.value_grid(VGT::range, ppid);
        auto calc_range = phys_view.make_calculator<RangeCalculator>(grid_id);
        real_type range = calc_range(energy);
        phys_view.dedx_range(range);

        return phys_view;
    }

  protected:
    std::shared_ptr<UrbanMscParams const> msc_params_;

    PhysicsStateStore physics_state_;
    ParticleStateStore particle_state_;
    GeoStateStore geo_state_;
    SimStateStore sim_state_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UrbanMscTest, coeff_data)
{
    auto mid = this->material()->find_material("G4_STAINLESS-STEEL");
    ASSERT_TRUE(mid);

    // Check MscMaterialDara for the current material (G4_STAINLESS-STEEL)
    UrbanMscMaterialData const& msc
        = msc_params_->host_ref().material_data[mid];

    EXPECT_SOFT_EQ(msc.coeffth1, 0.97326969977637379);
    EXPECT_SOFT_EQ(msc.coeffth2, 0.044188139325421663);
    EXPECT_SOFT_EQ(msc.d[0], 1.6889578380303167);
    EXPECT_SOFT_EQ(msc.d[1], 2.745018223507488);
    EXPECT_SOFT_EQ(msc.d[2], -2.2531516772497562);
    EXPECT_SOFT_EQ(msc.d[3], 0.052696806851297018);
    EXPECT_SOFT_EQ(msc.stepmin_a, 1e3 * 4.4449610414595817);
    EXPECT_SOFT_EQ(msc.stepmin_b, 1e3 * 1.5922149179564158);
    EXPECT_SOFT_EQ(msc.d_over_r, 0.64474963087322135);
    EXPECT_SOFT_EQ(msc.d_over_r_mh, 1.1248191999999999);
}

TEST_F(UrbanMscTest, msc_scattering)
{
    auto mid = this->material()->find_material("G4_STAINLESS-STEEL");
    ASSERT_TRUE(mid);

    // Test the step limitation algorithm and the msc sample scattering
    MscStep step_result;
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
    constexpr unsigned int nsamples = std::end(energy) - std::begin(energy);

    // Calculate range instead of hardcoding to ensure step and range values
    // are bit-for-bit identical when range limits the step. The first three
    // steps are not limited by range
    constexpr double step_is_range = -1;
    std::vector<double> step = {0.00279169, 0.412343, 0.0376414};
    step.resize(nsamples, step_is_range);

    for (unsigned int i : range(nsamples))
    {
        if (step[i] == step_is_range)
        {
            PhysicsTrackView phys = this->make_track_view(
                pdg::electron(), mid, MevEnergy{energy[i]});
            step[i] = phys.dedx_range();
        }
    }

    SimTrackView sim_track_view(sim_state_.ref(), ThreadId{0});
    sim_track_view = {};
    ParticleTrackView par_track_view{
        this->particle()->host_ref(), particle_state_.ref(), ThreadId{0}};
    GeoTrackView geo_view(
        this->geometry()->host_ref(), geo_state_.ref(), ThreadId{0});
    MaterialView material_view = this->material()->get(mid);

    ASSERT_EQ(nsamples, step.size());
    EXPECT_EQ(0, sim_track_view.num_steps());

    RandomEngine rng;
    std::vector<double> fstep;
    std::vector<double> angle;
    std::vector<double> displace;
    std::vector<char> action;
    Real3 direction{0, 0, 1};

    for (auto i : range(nsamples))
    {
        real_type r = i * 2 - real_type(1e-4);
        geo_view = {{r, r, r}, direction};

        MevEnergy inc_energy = MevEnergy{energy[i]};
        PhysicsTrackView phys = this->make_track_view(
            pdg::electron(), mid, inc_energy);

        UrbanMscStepLimit calc_limit(msc_params_->host_ref(),
                                     par_track_view,
                                     &phys,
                                     material_view.material_id(),
                                     geo_view.is_on_boundary(),
                                     geo_view.find_safety(),
                                     step[i]);

        step_result = calc_limit(rng);

        UrbanMscScatter scatter(msc_params_->host_ref(),
                                par_track_view,
                                &geo_view,
                                phys,
                                material_view,
                                step_result,
                                /* geo_limited = */ false);

        sample_result = scatter(rng);

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

    static double const expected_fstep[] = {0.00279169,
                                            0.15497550035228,
                                            0.0376414,
                                            0.078163867310103,
                                            0.0013704878213315,
                                            9.659931080008e-05,
                                            0.00074215629579835,
                                            3.1163160035577e-05};
    EXPECT_VEC_SOFT_EQ(expected_fstep, fstep);
    static double const expected_angle[] = {0.00031474130607916,
                                            0.79003683103898,
                                            -0.14560882721751,
                                            0,
                                            -0.32650640360665,
                                            0.013072020086723,
                                            0,
                                            0};
    EXPECT_VEC_NEAR(expected_angle, angle, 1e-10);
    static double const expected_displace[] = {8.19862035797085e-06,
                                               9.7530617641316e-05,
                                               -7.1670542039709e-05,
                                               0,
                                               -9.11378237130022e-05,
                                               9.78783890322556e-06,
                                               0,
                                               0};
    EXPECT_VEC_NEAR(expected_displace, displace, 1e-10);
    static char const expected_action[]
        = {'d', 'd', 'd', 'u', 'd', 'd', 'u', 'u'};
    EXPECT_VEC_EQ(expected_action, action);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
