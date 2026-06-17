//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/StateDependentLifecycleGlobal.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/ThreadId.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/g4/StateDependent.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class StateDependentLifecycleGlobalTest : public Test
{
};

TEST_F(StateDependentLifecycleGlobalTest, all)
{
    std::vector<std::string> lifecycles;
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
