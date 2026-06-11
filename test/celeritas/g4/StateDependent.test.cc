//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/StateDependent.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/g4/StateDependent.hh"

#include <memory>

#include "corecel/sys/ThreadId.hh"
#include "celeritas/ext/GeantSetup.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class StateDependentTest : public Test
{
};

TEST_F(StateDependentTest, all)
{
    std::vector<std::string> states;
    std::vector<std::string> lifecycles;
    std::unique_ptr<StateDependent> state_dep;
    state_dep = std::make_unique<StateDependent>(
        [&states, &state_dep](StreamId sid, GeantStateChange change) {
            EXPECT_EQ(StreamId{0}, sid);

            states.emplace_back(to_cstring(change));
            if (change == GeantStateChange::end_program)
            {
                state_dep.reset();
            }
        });
    // Deliberately leak: StateDependent deregisters itself on end_program, but
    // destroying it from inside its callback is unsafe.
    new StateDependent(
        [&lifecycles](StreamId sid, GeantStateChange change) {
            EXPECT_EQ(StreamId{0}, sid);

            lifecycles.emplace_back(to_cstring(change));
        },
        StateDependent::Mode::lifecycle);

    {
        GeantSetup setup(this->test_data_path("geocel", "lar-sphere.gdml"),
                         GeantPhysicsOptions{});
    }
    // See TrackingManagerTest for an example testing events as well
    static std::string const expected_states[] = {
        "begin_program",
        "initialize",
        "begin_init",
        "internal_init",
        "internal_init",
        "internal_init",
        "end_init",
        "begin_init",
        "end_init",
        "begin_init",
        "end_init",
        "begin_run",
        "end_run",
        "end_program",
    };
    EXPECT_VEC_EQ(expected_states, states);
    EXPECT_EQ(nullptr, state_dep);
    static std::string const expected_lifecycles[] = {
        "begin_run",
        "end_run",
        "end_program",
    };
    EXPECT_VEC_EQ(expected_lifecycles, lifecycles);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
