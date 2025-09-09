//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/PdfUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"

#include "Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the integral of a piecewise rectangular function.
 *
 * The value at the left point is taken for the interval.
 *
 * \todo Remove rectangular integrator (currently and likely always unused) and
 * possibly add higher-order integration methods if needed (e.g. for cross
 * section generation)
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
 * Estimate the mean and variance of a tabulated PDF.
 */
class MomentCalculator
{
  public:
    template<class T>
    struct Result
    {
        T mean{};
        T variance{};
    };

  public:
    //! Estimate the mean and variance
    template<class T>
    Result<T> operator()(Span<T const> x, Span<T const> f)
    {
        CELER_EXPECT(x.size() == f.size());
        CELER_EXPECT(x.size() >= 2);

        using Array2 = Array<T, 2>;

        TrapezoidSegmentIntegrator integrate;

        T integral{};
        T mean{};
        T variance{};
        Array2 prev{x[0], f[0]};
        for (auto i : range(std::size_t{1}, x.size()))
        {
            Array2 cur{x[i], f[i]};
            auto area = integrate(prev, cur);
            auto x_eval = T(0.5) * (cur[0] + prev[0]);
            integral += area;
            mean += area * x_eval;
            variance += area * ipow<2>(x_eval);
            prev = cur;
        }
        mean /= integral;
        variance = variance / integral - ipow<2>(mean);
        return {mean, variance};
    }
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
