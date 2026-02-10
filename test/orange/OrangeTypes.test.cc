//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTypes.test.cc
//---------------------------------------------------------------------------//
#include "orange/OrangeTypes.hh"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <string>
#include <type_traits>
#include <gtest/gtest.h>

#include "corecel/ScopedLogStorer.hh"
#include "corecel/Types.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(Tolerance, dbl)
{
    using TolT = Tolerance<double>;
    EXPECT_FALSE(TolT{});

    {
        SCOPED_TRACE("Default tolerance");
        auto const tol = TolT::from_default();
        EXPECT_TRUE(tol);
        EXPECT_SOFT_NEAR(
            std::sqrt(std::numeric_limits<double>::epsilon()), tol.rel, 0.1);
        EXPECT_SOFT_EQ(tol.rel, tol.abs);
        EXPECT_SOFT_EQ(1.5e-8, tol.rel);
    }
    {
        SCOPED_TRACE("Tolerance with other length scale");
        auto const tol = Tolerance<double>::from_default(1e-4);
        EXPECT_SOFT_EQ(1.5e-8, tol.rel);
        EXPECT_SOFT_EQ(1.5e-12, tol.abs);
        EXPECT_SOFT_EQ(1e-10, ipow<2>(Tolerance<double>::sqrt_quadratic()));
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

TEST(Tolerance, single)
{
    using TolT = Tolerance<float>;

    {
        SCOPED_TRACE("Default tolerance");
        auto const tol = TolT::from_default();
        EXPECT_TRUE(tol);
        EXPECT_SOFT_NEAR(
            std::sqrt(std::numeric_limits<float>::epsilon()), tol.rel, 0.1);
        EXPECT_SOFT_EQ(tol.rel, tol.abs);
        EXPECT_SOFT_EQ(0.0003f, tol.rel);
    }
    {
        ScopedLogStorer scoped_log_{&celeritas::world_logger(),
                                    LogLevel::warning};
        auto tol = TolT::from_relative(1e-9f);
        EXPECT_GT(tol.rel, 1e-9f);
        static char const* const expected_log_messages[]
            = {"Clamped relative tolerance 1e-9 to machine epsilon 1.192e-7"};
        EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
        static char const* const expected_log_levels[] = {"warning"};
        EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
    }
    {
        ScopedLogStorer scoped_log_{&celeritas::world_logger(),
                                    LogLevel::warning};
        TolT tol;
        tol.rel = 1e-9f;
        tol.abs = 1e-40f;
        auto c = tol.clamped();
        EXPECT_SOFT_EQ(0, c.rel);
        EXPECT_SOFT_EQ(0, c.abs);
        EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
    }
}

TEST(Zorder, round_trip)
{
    // Test round-tripping of zorder
    for (auto zo : {ZOrder::invalid,
                    ZOrder::background,
                    ZOrder::media,
                    ZOrder::array,
                    ZOrder::hole,
                    ZOrder::implicit_exterior,
                    ZOrder::exterior})
    {
        EXPECT_EQ(zo, to_zorder(to_char(zo)));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
