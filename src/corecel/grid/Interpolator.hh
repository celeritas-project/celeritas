//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/Interpolator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

#include "GridTypes.hh"

#include "detail/InterpolatorTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interpolate, with either linear or log in x and y.
 *
 * \tparam XI Transform to apply to X coordinate
 * \tparam YI Transform to apply to Y coordinate
 * \tparam T  Floating point type
 *
 * The inputs are given as two (x,y) pairs. The same two pairs can be used to
 * interpolate successively with this functor.
 *
 * Interpolation on the transformed coordinates is the solution for
 * \f$ y \f$ in \f[
 \frac{f_y(y) - f_y(y_l)}{f_y(y_r) - f_y(y_l)}
 = \frac{f_x(x) - f_x(x_l)}{f_x(x_r) - f_x(x_l)}
 \f]
 *
 * where \f$ f_d(v) = v \f$ for linear interpolation on the \f$ d \f$ axis
 * and \f$ f_d(v) = \log v \f$ for log interpolation.
 *
 * Solving for \f$y\f$, the interpolation can be rewritten to minimize
 * transcendatal operations and efficiently perform multiple interpolations
 * over the same point range:
 * \f[
  y = f^{-1}_y \left( f_y(y_l) + \frac{ p_y(n_y(y_l), y_r) }{ p_x(n_x(x_l),
 x_r)} \times p_x(n_x(x_l), x) \right)
 \f]

 * where transformed addition is \f$p_y(y_1, y_2) \equiv f_y(y_1) + f_y(y_2)\f$
 ,
 * transformed negation is  \f$n_y(y_1) \equiv f^{-1}_y( - f_y(y_1) )\f$
 * and \f$ f_y(y) = y \f$ for linear interpolation in y
 * and \f$ f_y(y) = \log y \f$ for log interpolation in y.
 *
 * Instantiating the interpolator precalculates the transformed intercept and
 * slope terms, as well as the negated x-left term.
 * At each evaluation of the instantiated Interpolator, only the
 * inverse-transform and add-transformed operation need be applied.
 */
template<Interp XI = Interp::linear,
         Interp YI = Interp::linear,
         typename T = ::celeritas::real_type>
class Interpolator
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = T;
    using Point = Array<T, 2>;
    //!@}

  public:
    // Construct with left and right values for x and y
    inline CELER_FUNCTION Interpolator(Point left, Point right);

    // Interpolate
    inline CELER_FUNCTION real_type operator()(real_type x) const;

  private:
    real_type intercept_;  //!> f_y(y_l)
    real_type slope_;  //!> ratio of g(y) to g(x)
    real_type offset_;  //!> n_x(x_l)

    using XTraits_t = detail::InterpolatorTraits<XI, real_type>;
    using YTraits_t = detail::InterpolatorTraits<YI, real_type>;
};

//! Linear interpolation
template<class T>
using LinearInterpolator = Interpolator<Interp::linear, Interp::linear, T>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with left and right values for x and y.
 */
template<Interp XI, Interp YI, class T>
CELER_FUNCTION Interpolator<XI, YI, T>::Interpolator(Point left, Point right)
{
    enum
    {
        X = 0,
        Y = 1
    };

    CELER_EXPECT(left[X] < right[X]);
    CELER_EXPECT(XTraits_t::valid_domain(left[X]));
    CELER_EXPECT(XTraits_t::valid_domain(right[X]));
    CELER_EXPECT(YTraits_t::valid_domain(left[Y]));
    CELER_EXPECT(YTraits_t::valid_domain(right[Y]));

    intercept_ = YTraits_t::transform(left[Y]);
    slope_ = (YTraits_t::add_transformed(
                  YTraits_t::negate_transformed(left[Y]), right[Y])
              / XTraits_t::add_transformed(
                  XTraits_t::negate_transformed(left[X]), right[X]));
    offset_ = XTraits_t::negate_transformed(left[X]);
    CELER_ENSURE(!std::isnan(intercept_) && !std::isnan(slope_)
                 && !std::isnan(offset_));
}

//---------------------------------------------------------------------------//
/*!
 * Interpolate linearly on the transformed type.
 */
template<Interp XI, Interp YI, class T>
CELER_FUNCTION auto Interpolator<XI, YI, T>::operator()(real_type x) const
    -> real_type
{
    CELER_EXPECT(XTraits_t::valid_domain(x));
    real_type result = YTraits_t::transform_inv(
        std::fma(slope_, XTraits_t::add_transformed(offset_, x), intercept_));

    CELER_ENSURE(!std::isnan(result));
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
