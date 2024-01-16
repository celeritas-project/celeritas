//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/msc/detail/UrbanPositronCorrector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate a correction on theta0 for positrons.
 */
class UrbanPositronCorrector
{
  public:
    // Construct with effective Z
    explicit inline CELER_FUNCTION UrbanPositronCorrector(real_type zeff);

    // Calculate correction with the ratio of avg energy to electron mass
    inline CELER_FUNCTION real_type operator()(real_type y) const;

  private:
    real_type a_, b_, c_, d_, mult_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with effective Z.
 */
CELER_FUNCTION UrbanPositronCorrector::UrbanPositronCorrector(real_type zeff)
{
    CELER_EXPECT(zeff >= 1);
    using PolyLin = PolyEvaluator<real_type, 1>;
    using PolyQuad = PolyEvaluator<real_type, 2>;

    a_ = PolyLin(0.994, -4.08e-3)(zeff);
    b_ = PolyQuad(7.16, 52.6, 365)(1 / zeff);
    c_ = PolyLin(1, -4.47e-3)(zeff);
    d_ = real_type(1.21e-3) * zeff;
    mult_ = PolyQuad(1.41125, -1.86427e-2, 1.84035e-4)(zeff);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the correction.
 *
 * The input parameter is a unitless ratio of the geometric mean energy to the
 * electron mass : \f[
   y = \frac{\sqrt{E_i E_f}}{m_e}
  \f]
 */
CELER_FUNCTION real_type UrbanPositronCorrector::operator()(real_type y) const
{
    CELER_EXPECT(y > 0);
    real_type corr = [this, y] {
        constexpr real_type xl = 0.6;
        constexpr real_type xh = 0.9;
        constexpr real_type e = 113;

        // x = sqrt((y^2 + 2y) / (y^2 + 2y + 1))
        real_type x = std::sqrt(y * (y + 2) / ipow<2>(y + 1));
        if (x < xl)
        {
            return a_ * (1 - std::exp(-b_ * x));
        }
        if (x > xh)
        {
            return c_ + d_ * std::exp(e * (x - 1));
        }
        real_type yl = a_ * (1 - std::exp(-b_ * xl));
        real_type yh = c_ + d_ * std::exp(e * (xh - 1));
        real_type y0 = (yh - yl) / (xh - xl);
        real_type y1 = yl - y0 * xl;
        return y0 * x + y1;
    }();

    return corr * mult_;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
