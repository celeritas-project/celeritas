//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/SplitMix64.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/engine/SplitMix64.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(SplitMix64Test, host)
{
    celeritas::SplitMix64 sm(12345);

    EXPECT_EQ(2454886589211414944ul, sm());
    EXPECT_EQ(3778200017661327597ul, sm());
    EXPECT_EQ(2205171434679333405ul, sm());
    EXPECT_EQ(3248800117070709450ul, sm());
    EXPECT_EQ(9350289611492784363ul, sm());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
