//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ScopedStreamRedirect.test.cc
//---------------------------------------------------------------------------//
#include "base/ScopedStreamRedirect.hh"

#include "celeritas_test.hh"

using celeritas::ScopedStreamRedirect;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ScopedStreamRedirectTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ScopedStreamRedirectTest, all)
{
    const auto* orig_buf = std::cout.rdbuf();
    {
        ScopedStreamRedirect redirect(&std::cout);
        EXPECT_NE(orig_buf, std::cout.rdbuf());

        std::cout << "More output  \n";
        EXPECT_EQ("More output", redirect.str());
    }
    EXPECT_EQ(orig_buf, std::cout.rdbuf());
}
