//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/CdfUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the integral of a piecewise rectangular function.
 *
 * The value at the left point is taken for the interval.
 */
struct PostRectangleSegmentIntegrator
{
    template<class T>
    T operator()(Array<T, 2> lo, Array<T, 2> hi) const
    {
        return (hi[0] - lo[0]) * lo[1];
    }
};

//---------------------------------------------------------------------------//
/*!
 * Calculate the integral of a piecewise linear function.
 */
struct TrapezoidSegmentIntegrator
{
    template<class T>
    T operator()(Array<T, 2> lo, Array<T, 2> hi) const
    {
        return T(0.5) * (hi[0] - lo[0]) * (hi[1] + lo[1]);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Integrate a piecewise function.
 *
 * To construct a CDF, `init` should be zero, and the destination should be
 * normalized by its final value afterward.
 */
template<class I>
class SegmentIntegrator
{
  public:
    //! Construct with integrator
    explicit SegmentIntegrator(I&& integrate)
        : integrate_{std::forward<I>(integrate)}
    {
    }

    //! Integrate a function
    template<class T, std::size_t N, std::size_t M>
    void operator()(Span<T const, N> x,
                    Span<T const, N> f,
                    Span<T, M> dst,
                    T init = {})
    {
        CELER_EXPECT(x.size() == f.size());
        CELER_EXPECT(x.size() == dst.size());

        using Array2 = Array<T, 2>;

        Array2 prev{x[0], f[0]};
        dst[0] = init;
        for (auto i : range(std::size_t{1}, x.size()))
        {
            Array2 cur{x[i], f[i]};
            init += integrate_(prev, cur);
            dst[i] = init;
            prev = cur;
        }
    }

  private:
    I integrate_;
};

//---------------------------------------------------------------------------//
/*!
 * Normalize a vector by the final value and check for monotonicity.
 */
template<class T, std::size_t N>
inline void normalize_cdf(Span<T, N> x)
{
    CELER_EXPECT(!x.empty());
    CELER_EXPECT(x.back() > 0);
    T norm{1 / x.back()};
    for (auto i : range(x.size() - 1))
    {
        CELER_ASSERT(x[i + 1] >= x[i]);
        x[i] *= norm;
    }
    x.back() = 1;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
