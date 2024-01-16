//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedStreamRedirect.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/ScopedStreamRedirect.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(ScopedStreamRedirectTest, all)
{
    auto const* orig_buf = std::cout.rdbuf();
    {
        ScopedStreamRedirect redirect(&std::cout);
        EXPECT_NE(orig_buf, std::cout.rdbuf());

        std::cout << "More output  \n";
        EXPECT_EQ("More output", redirect.str());
    }
    EXPECT_EQ(orig_buf, std::cout.rdbuf());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
