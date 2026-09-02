//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/StateDependent.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/g4/StateDependent.hh"

#include <G4StateManager.hh>

#include "corecel/sys/ThreadId.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
void set_geant_state(G4ApplicationState state)
{
    auto* sm = G4StateManager::GetStateManager();
    CELER_ASSERT(sm);
    EXPECT_TRUE(sm->SetNewState(state));
}

//---------------------------------------------------------------------------//
void run_lifecycle()
{
    set_geant_state(G4State_GeomClosed);
    set_geant_state(G4State_Idle);
    set_geant_state(G4State_Quit);
}

//---------------------------------------------------------------------------//
}  // namespace

class StateDependentTest : public Test
{
};

TEST_F(StateDependentTest, raw)
{
    set_geant_state(G4State_Idle);
    std::vector<std::string> states;
    // Deliberately leak raw observers: StateDependent deregisters itself on
    // end_program, but destroying it from inside its callback is unsafe.
    new StateDependent([&states](StreamId sid, GeantStateChange change) {
        EXPECT_EQ(StreamId{0}, sid);

        states.emplace_back(to_cstring(change));
    });

    static std::string const expected_states[] = {
        "begin_run",
        "end_run",
        "end_program",
    };
    run_lifecycle();
    EXPECT_VEC_EQ(expected_states, states);
}

TEST_F(StateDependentTest, lifecycle_global)
{
    set_geant_state(G4State_Idle);
    std::vector<std::string> lifecycles;
    // Deliberately leak: StateDependent deregisters itself on end_program, but
    // destroying it from inside its callback is unsafe.
    new StateDependent(
        [&lifecycles](StreamId sid, GeantStateChange change) {
            EXPECT_EQ(StreamId{0}, sid);

            lifecycles.emplace_back(to_cstring(change));
        },
        StateDependent::Mode::lifecycle);

    static std::string const expected_lifecycles[] = {
        "begin_run",
        "end_run",
        "end_program",
    };
    run_lifecycle();
    EXPECT_VEC_EQ(expected_lifecycles, lifecycles);
}

TEST_F(StateDependentTest, lifecycle_local)
{
    set_geant_state(G4State_Idle);
    std::vector<std::string> lifecycles;
    // Deliberately leak: StateDependent deregisters itself on end_program, but
    // destroying it from inside its callback is unsafe.
    new StateDependent(
        [&lifecycles](StreamId sid, GeantStateChange change) {
            EXPECT_EQ(StreamId{0}, sid);

            lifecycles.emplace_back(to_cstring(change));
        },
        StateDependent::Mode::lifecycle,
        StateDependent::LifecycleRole::local);

    static std::string const expected_lifecycles[] = {
        "begin_run",
        "end_run",
    };
    run_lifecycle();
    EXPECT_VEC_EQ(expected_lifecycles, lifecycles);
}

TEST_F(StateDependentTest, lifecycle_local_terminal_cleanup)
{
    set_geant_state(G4State_Idle);
    std::vector<std::string> lifecycles;
    // Deliberately leak: StateDependent deregisters itself on end_program, but
    // destroying it from inside its callback is unsafe.
    new StateDependent(
        [&lifecycles](StreamId sid, GeantStateChange change) {
            EXPECT_EQ(StreamId{0}, sid);

            lifecycles.emplace_back(to_cstring(change));
        },
        StateDependent::Mode::lifecycle,
        StateDependent::LifecycleRole::local);

    set_geant_state(G4State_GeomClosed);
    set_geant_state(G4State_Quit);

    static std::string const expected_lifecycles[] = {
        "begin_run",
        "end_run",
    };
    EXPECT_VEC_EQ(expected_lifecycles, lifecycles);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
