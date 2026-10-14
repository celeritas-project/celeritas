//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopeExit.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/ScopeExit.hh"

#include <stdexcept>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(ScopeExitTest, basic)
{
    int count = 0;
    {
        ScopeExit cleanup{[&count] { ++count; }};
        EXPECT_EQ(0, count);
    }
    EXPECT_EQ(1, count);
}

TEST(ScopeExitTest, exception)
{
    int count = 0;
    EXPECT_THROW(
        {
            ScopeExit cleanup{[&count] { ++count; }};
            throw std::runtime_error{"error"};
        },
        std::runtime_error);
    EXPECT_EQ(1, count);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
