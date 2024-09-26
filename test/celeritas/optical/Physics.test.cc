//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Physics.test.cc
//---------------------------------------------------------------------------//
#include <set>
#include <vector>

#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/Model.hh"
#include "celeritas/optical/PhysicsParams.hh"
#include "celeritas/optical/PhysicsStepView.hh"
#include "celeritas/optical/PhysicsTrackView.hh"

#include "OpticalMockTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
using Energy = celeritas::units::MevEnergy;

//---------------------------------------------------------------------------//
// PhysicsParamsTest
//---------------------------------------------------------------------------//

class PhysicsParamsTest : public OpticalMockTestBase
{
  protected:
    void SetUp() override {}
};

TEST_F(PhysicsParamsTest, scalars)
{
    PhysicsParamsScalars const& scalars
        = this->optical_physics()->host_ref().scalars;

    // Scalars should be assigned and valid
    EXPECT_TRUE(scalars);

    // Expect scalars num models to be 4
    EXPECT_EQ(4, scalars.num_models);
}

TEST_F(PhysicsParamsTest, models)
{
    PhysicsParams const& physics = *this->optical_physics();

    // Expect 4 mock models to be built and present
    EXPECT_EQ(4, physics.num_models());

    // Each model should have correct name, description, and action ID
    std::vector<std::string_view> model_names;
    std::vector<std::string_view> model_desc;
    std::set<ActionId> action_ids;

    for (auto mid : range(ModelId{physics.num_models()}))
    {
        ASSERT_TRUE(physics.model(mid));

        auto const& model = *physics.model(mid);

        model_names.push_back(model.label());
        model_desc.push_back(model.description());
        action_ids.insert(model.action_id());
    }

    static std::string_view const expected_model_names[] = {""};
    EXPECT_VEC_EQ(expected_model_names, model_names);

    static std::string_view const expected_model_desc[] = {""};
    EXPECT_VEC_EQ(expected_model_desc, model_desc);

    // ActionId range should correspond to model actions
    EXPECT_EQ(4, action_ids.size());
    for (auto action_id : physics.model_actions())
    {
        EXPECT_EQ(1, action_ids.count(action_id));
    }
}

//---------------------------------------------------------------------------//
// PhysicsViewHostTest
//---------------------------------------------------------------------------//

class PhysicsViewHostTest : public PhysicsParamsTest
{
  protected:
    //!@{
    //! \name Type aliases
    using StateStore = CollectionStateStore<PhysicsStateData, MemSpace::host>;
    using ParamsHostRef = HostCRef<PhysicsParamsData>;
    //!@}

    void SetUp() override
    {
        PhysicsParamsTest::SetUp();

        CELER_ASSERT(this->optical_physics());
        params_ref = this->optical_physics()->host_ref();
        state = StateStore(params_ref, num_tracks);
    }

    PhysicsTrackView
    make_track_view(OpticalMaterialId mat_id, TrackSlotId tid) const
    {
        CELER_EXPECT(mat_id && tid && tid.get() < num_tracks);

        PhysicsTrackView phys(params_ref, state.ref(), mat_id, tid);
        phys = PhysicsTrackInitializer{};

        return phys;
    }

    PhysicsStepView make_step_view(TrackSlotId tid) const
    {
        CELER_EXPECT(tid && tid.get() < num_tracks);
        return PhysicsStepView(params_ref, state.ref(), tid);
    }

    StateStore state;
    ParamsHostRef params_ref;

    size_type const num_tracks = 5;
};

TEST_F(PhysicsViewHostTest, track_parameters)
{
    // Use an arbitrary track view - these are track independent parameters
    auto track_view
        = this->make_track_view(OpticalMaterialId{1}, TrackSlotId{2});

    // Num optical models should match physics
    EXPECT_EQ(this->optical_physics()->num_models(),
              track_view.num_optical_models());

    // action to model -> model to action should be identity
    for (auto model_id : range(ModelId{track_view.num_optical_models()}))
    {
        auto action_id = track_view.model_to_action(model_id);
        EXPECT_EQ(model_id, track_view.action_to_model(action_id));
    }
}

TEST_F(PhysicsViewHostTest, track_interaction_mfp)
{
    for (auto tid : range(TrackSlotId{num_tracks}))
    {
        // interaction MFP should be material independent - make arbitrary
        // material ID for testing
        auto phys = this->make_track_view(
            OpticalMaterialId{tid.get()
                              % this->optical_material()->num_materials()},
            tid);

        // Newly initialized tracks should not have MFPs
        EXPECT_FALSE(phys.has_interaction_mfp());

        // Set interaction MFP
        real_type mfp = tid.get() + 1.234;
        phys.interaction_mfp(mfp);

        //  - has MFP should be true
        EXPECT_TRUE(phys.has_interaction_mfp());

        //  - should have same MFP via accessor
        EXPECT_EQ(mfp, phys.interaction_mfp());

        // Reset interaction MFP
        phys.reset_interaction_mfp();

        //  - has MFP should be false
        EXPECT_FALSE(phys.has_interaction_mfp());
    }
}

TEST_F(PhysicsViewHostTest, track_mfp_grid)
{
    static int expected_grid_ids[] = {-1};

    for (auto tid : range(TrackSlotId{num_tracks}))
    {
        std::vector<int> grid_ids;

        for (auto mat_id : range(
                 OpticalMaterialId{this->optical_material()->num_materials()}))
        {
            auto phys = this->make_track_view(mat_id, tid);

            for (auto model_id : range(ModelId{phys.num_optical_models()}))
            {
                auto grid_id = phys.mfp_grid(model_id);
                grid_ids.push_back(grid_id ? static_cast<int>(grid_id.get())
                                           : -1);
            }
        }

        EXPECT_VEC_EQ(expected_grid_ids, grid_ids);
    }
}

TEST_F(PhysicsViewHostTest, track_calc_mfp)
{
    Energy const energy{1.0};

    static real_type expected_mfps[] = {0};

    for (auto tid : range(TrackSlotId{num_tracks}))
    {
        std::vector<real_type> mfps;

        for (auto mat_id : range(
                 OpticalMaterialId{this->optical_material()->num_materials()}))
        {
            auto phys = this->make_track_view(mat_id, tid);

            for (auto model_id : range(ModelId{phys.num_optical_models()}))
            {
                mfps.push_back(phys.calc_mfp(model_id, energy));
            }
        }

        EXPECT_VEC_EQ(expected_mfps, mfps);
    }
}

TEST_F(PhysicsViewHostTest, step_xs)
{
    // Construct mock micro xs data
    std::vector<real_type> expected_micro_xs;
    for (unsigned int i = 0;
         i < num_tracks * this->optical_physics()->num_models();
         i++)
    {
        expected_micro_xs.push_back(3.14 * static_cast<real_type>(i));
    }

    // Fill each track micro xs scratch data and total xs with mock data
    for (auto tid : range(TrackSlotId{num_tracks}))
    {
        auto step = this->make_step_view(tid);

        EXPECT_EQ(real_type{0}, step.macro_xs());

        real_type total_xs = 0;
        for (auto mid : range(ModelId{this->optical_physics()->num_models()}))
        {
            real_type xs
                = expected_micro_xs[mid.get() * num_tracks + tid.get()];
            step.per_model_xs(mid) = xs;
            total_xs += xs;
        }

        step.macro_xs(total_xs);
        EXPECT_EQ(total_xs, step.macro_xs());
    }

    // Check that there were no overwrites of the scratch data
    for (auto tid : range(TrackSlotId{num_tracks}))
    {
        auto const step = this->make_step_view(tid);

        real_type total_xs = 0;
        for (auto mid : range(ModelId{this->optical_physics()->num_models()}))
        {
            real_type xs
                = expected_micro_xs[mid.get() * num_tracks + tid.get()];
            EXPECT_EQ(xs, step.per_model_xs(mid));
            total_xs += xs;
        }

        EXPECT_EQ(total_xs, step.macro_xs());
    }
}

TEST_F(PhysicsViewHostTest, step_energy_deposit)
{
    // Create mock energy deposition data
    std::vector<Energy> expected_energy_deposits;
    for (unsigned int i = 0; i < num_tracks; i++)
    {
        expected_energy_deposits.push_back(Energy{2.5 * real_type{i}});
    }

    // Reset then assign mock energy deposition
    for (auto tid : range(TrackSlotId{num_tracks}))
    {
        auto step = this->make_step_view(tid);

        step.reset_energy_deposition();
        EXPECT_EQ(Energy{0}, step.energy_deposition());

        step.deposit_energy(expected_energy_deposits[tid.get()]);
    }

    // Check there were no overwrites of the energy depositions
    for (auto tid : range(TrackSlotId{num_tracks}))
    {
        auto const step = this->make_step_view(tid);

        EXPECT_EQ(expected_energy_deposits[tid.get()],
                  step.energy_deposition());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
