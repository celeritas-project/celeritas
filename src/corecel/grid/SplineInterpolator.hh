//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/SplineInterpolator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/PolyEvaluator.hh"

#include "detail/InterpolatorTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interpolate using a cubic spline.
 *
 * Given a set of \f$ n \f$ data points \f$ (x_i, y_i) \f$ such that \f$ x_0 <
 * x_1 < \dots < x_{n - 1} \f$, a cubic spline \f$ S(x) \f$ interpolating on
 * the points is a piecewise polynomial function consisting of \f$ n - 1 \f$
 * cubic polynomials \f$ S_i \f$ defined on \f$ [x_i, x_{i + 1}] \f$. The \f$
 * S_i \f$ are joined at \f$ x_i \f$ such that both the first and second
 * derivatives, \f$ S'_i \f$ and \f$ S''_i \f$, are continuous.
 *
 * The \f$ i^{\text{th}} \f$ piecewise polynomial \f$ S_i \f$ is given by:
 * \f[
   S_i(x) = a_0 + a_1(x - x_i) + a_2(x - x_i)^2 + a_3(x - x_i)^3,
 * \f]
 * where \f$ a_i \f$ are the polynomial coefficients, expressed in terms of the
 * second derivatives as:
 * \f[
 * \begin{aligned}
   a_0 &= y_i \\
   a_1 &= \frac{\Delta y_i}{\Delta x_i} - \frac{\Delta x_i}{6}
          \left[ S''_{i + 1} + 2 S''_{i} \right] \\
   a_2 &= \frac{S''_i}{2} \\
   a_3 &= \frac{1}{6 \Delta x_i} \left[ S''_{i + 1} - S''_i \right]
 * \end{aligned}
 * \f]
 */
template<typename T = ::celeritas::real_type>
class SplineInterpolator
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = T;
    //!@}

    //! (x, y) point and second derivative
    struct Spline
    {
        real_type x;
        real_type y;
        real_type ddy;
    };

  public:
    // Construct with left and right values for x, y, and the second derivative
    inline CELER_FUNCTION SplineInterpolator(Spline left, Spline right);

    // Interpolate
    inline CELER_FUNCTION real_type operator()(real_type x) const;

  private:
    real_type x_lower_;  //!< Lower grid point \f$ x_i \f$
    Array<T, 4> a_;  //!< Spline coefficients
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with left and right values for x, y, and the second derivative.
 */
template<class T>
CELER_FUNCTION
SplineInterpolator<T>::SplineInterpolator(Spline left, Spline right)
{
    CELER_EXPECT(left.x < right.x);

    x_lower_ = left.x;
    real_type h = right.x - left.x;
    a_[0] = left.y;
    a_[1] = (right.y - left.y) / h - h / 6 * (right.ddy + 2 * left.ddy);
    a_[2] = T(0.5) * left.ddy;
    a_[3] = 1 / (6 * h) * (right.ddy - left.ddy);
}

//---------------------------------------------------------------------------//
/*!
 * Interpolate using a cubic spline.
 */
template<class T>
CELER_FUNCTION auto SplineInterpolator<T>::operator()(real_type x) const
    -> real_type
{
    return PolyEvaluator{a_}(x - x_lower_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
