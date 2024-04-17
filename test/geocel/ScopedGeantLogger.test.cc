//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/ScopedGeantLogger.test.cc
//---------------------------------------------------------------------------//
#include "geocel/ScopedGeantLogger.hh"

#include <G4ios.hh>

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class ScopedGeantLoggerTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(ScopedGeantLoggerTest, host)
{
    ScopedGeantLogger scoped_g4;
    G4cout << "This is not be captured by scoped logger" << endl;

    ScopedLogStorer scoped_log_{&celeritas::world_logger(), LogLevel::debug};
    G4cout << "Standard output" << endl;
    G4cerr << "Standard err" << endl;
    G4cerr << "WARNING - nub nub" << endl;
    G4cout << "warning: from cout" << endl;
    G4cerr << "ERROR - derpaderp" << endl;
    G4cout << "G4Material warning: things are bad" << endl;

    static char const* const expected_log_messages[] = {
        "Standard output",
        "Standard err",
        "nub nub",
        "from cout",
        "derpaderp",
        "things are bad",
    };
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[]
        = {"diagnostic", "info", "warning", "warning", "error", "warning"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

TEST_F(ScopedGeantLoggerTest, nesting)
{
    ScopedGeantLogger scoped_g4;
    {
        ScopedGeantLogger scoped_g4;
        {
            ScopedGeantLogger scoped_g4;
            ScopedLogStorer scoped_log_{&celeritas::world_logger(),
                                        LogLevel::debug};
            G4cout << "This should still work" << endl;
            static char const* const expected_log_messages[]
                = {"This should still work"};
            EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
            static char const* const expected_log_levels[] = {"diagnostic"};
            EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
