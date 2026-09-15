//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/Types.test.cc
//---------------------------------------------------------------------------//
#include "geocel/Types.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(GeoStatusTest, validity)
{
    EXPECT_FALSE(is_valid(GeoStatus::error));
    EXPECT_FALSE(is_valid(GeoStatus::invalid));
    EXPECT_TRUE(is_valid(GeoStatus::interior));
    EXPECT_TRUE(is_valid(GeoStatus::boundary_inc));
    EXPECT_TRUE(is_valid(GeoStatus::boundary_out));
}

TEST(GeoStatusTest, boundary)
{
    EXPECT_FALSE(is_on_boundary(GeoStatus::error));
    EXPECT_FALSE(is_on_boundary(GeoStatus::invalid));
    EXPECT_FALSE(is_on_boundary(GeoStatus::interior));
    EXPECT_TRUE(is_on_boundary(GeoStatus::boundary_inc));
    EXPECT_TRUE(is_on_boundary(GeoStatus::boundary_out));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
