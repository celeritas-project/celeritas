//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/Types.test.cc
//---------------------------------------------------------------------------//
#include "corecel/Types.hh"

#include <type_traits>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using namespace celeritas::literals;

TEST(TypesTest, real_literal)
{
    constexpr auto half = 0.5_r;
    constexpr auto two = 2_r;

    EXPECT_TRUE((std::is_same_v<real_type, decltype(0.5_r)>));
    EXPECT_TRUE((std::is_same_v<real_type, decltype(2_r)>));
    EXPECT_EQ(static_cast<real_type>(0.5), half);
    EXPECT_EQ(static_cast<real_type>(2), two);
}

TEST(TypesTest, integer_literal)
{
    constexpr auto half = 0_sz;
    constexpr auto two = 2_sz;

    EXPECT_TRUE((std::is_same_v<size_type, decltype(0_sz)>));
    EXPECT_TRUE((std::is_same_v<size_type, decltype(2_sz)>));
    EXPECT_EQ(static_cast<size_type>(0), half);
    EXPECT_EQ(static_cast<size_type>(2), two);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
