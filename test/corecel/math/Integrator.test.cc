//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Integrator.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/Integrator.hh"

#include <cmath>
#include <utility>

#include "celeritas_test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class F>
struct DiagnosticFunc
{
    explicit DiagnosticFunc(F&& eval) : eval_{std::forward<F>(eval)} {}

    //! Evaluate the underlying function and increment the counter
    real_type operator()(real_type v)
    {
        ++count_;
        return eval_(v);
    }

    //! Get and reset the counter
    size_type exchange_count() { return std::exchange(count_, 0); }

    F eval_;
    size_type count_{0};
};

TEST(IntegratorTest, constant)
{
    DiagnosticFunc f{[](real_type) { return 10; }};
    {
        Integrator integrate{f};
        EXPECT_SOFT_EQ(10.0, integrate(1, 2));
        EXPECT_EQ(3, f.exchange_count());
        EXPECT_SOFT_EQ(10 * 10.0, integrate(2, 12));
        EXPECT_EQ(3, f.exchange_count());
    }
}

TEST(IntegratorTest, linear)
{
    DiagnosticFunc f{[](real_type x) { return 2 * x; }};
    {
        Integrator integrate{f};
        EXPECT_SOFT_EQ(4 - 1, integrate(1, 2));
        EXPECT_EQ(3, f.exchange_count());
        EXPECT_SOFT_EQ(16 - 4, integrate(2, 4));
        EXPECT_EQ(3, f.exchange_count());
    }
}

TEST(IntegratorTest, quadratic)
{
    DiagnosticFunc f{[](real_type x) { return 3 * ipow<2>(x); }};
    {
        real_type const eps = IntegratorOptions{}.epsilon;
        Integrator integrate{f};
        EXPECT_SOFT_NEAR(8 - 1, integrate(1, 2), eps);
        EXPECT_EQ(17, f.exchange_count());
        EXPECT_SOFT_NEAR(64 - 8, integrate(2, 4), eps);
        EXPECT_EQ(17, f.exchange_count());
    }
    {
        IntegratorOptions opts;
        opts.epsilon = 1e-5;
        Integrator integrate{f, opts};
        EXPECT_SOFT_NEAR(8 - 1, integrate(1, 2), opts.epsilon);
        EXPECT_EQ(257, f.exchange_count());
        EXPECT_SOFT_NEAR(64 - 8, integrate(2, 4), opts.epsilon);
        EXPECT_EQ(257, f.exchange_count());
    }
}

TEST(IntegratorTest, gauss)
{
    DiagnosticFunc f{
        [](real_type r) { return ipow<2>(r) * std::exp(-ipow<2>(r)); }};
    {
        Integrator integrate{f};
        EXPECT_SOFT_EQ(0.057594067180233119, integrate(0, 0.597223));
        EXPECT_EQ(33, f.exchange_count());
        EXPECT_SOFT_EQ(0.16739988271111467, integrate(0.597223, 1.09726));
        EXPECT_EQ(17, f.exchange_count());
        EXPECT_SOFT_EQ(0.20618863449804861, integrate(1.09726, 2.14597));
        EXPECT_EQ(5, f.exchange_count());
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
