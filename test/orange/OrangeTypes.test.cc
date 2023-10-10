//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTypes.test.cc
//---------------------------------------------------------------------------//
#include "orange/OrangeTypes.hh"

#include "celeritas_test.hh"
// #include "OrangeTypes.test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class OrangeTypesTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(OrangeTypesTest, tolerances)
{
    using TolT = Tolerance<>;
    EXPECT_FALSE(TolT{});

    EXPECT_SOFT_EQ(1e-10, ipow<2>(TolT::sqrt_quadratic()));

    {
        SCOPED_TRACE("Default tolerance");
        auto const tol = TolT::from_default();
        EXPECT_TRUE(tol);
        EXPECT_SOFT_NEAR(
            std::sqrt(std::numeric_limits<real_type>::epsilon()), tol.rel, 0.5);
        EXPECT_SOFT_EQ(tol.rel, tol.abs);
        if constexpr (std::is_same_v<real_type, double>)
        {
            EXPECT_SOFT_EQ(1e-8, tol.rel);
        }
    }
    {
        SCOPED_TRACE("Tolerance with other length scale");
        auto const tol = TolT::from_default(1e-4);
        EXPECT_SOFT_EQ(1e-8, tol.rel);
        EXPECT_SOFT_EQ(1e-12, tol.abs);
    }
    {
        SCOPED_TRACE("Tolerance with arbitrary relative");
        auto const tol = TolT::from_relative(1e-5);
        EXPECT_SOFT_EQ(1e-5, tol.rel);
        EXPECT_SOFT_EQ(1e-5, tol.abs);
    }
    {
        SCOPED_TRACE("Tolerance with arbitrary relative and length scale");
        auto const tol = TolT::from_relative(1e-5, 0.1);
        EXPECT_SOFT_EQ(1e-5, tol.rel);
        EXPECT_SOFT_EQ(1e-6, tol.abs);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
