//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Interpolator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Types.hh"
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
template<Interp XI  = Interp::linear,
         Interp YI  = Interp::linear,
         typename T = ::celeritas::real_type>
class Interpolator
{
  public:
    //!@{
    //! Public type aliases
    using real_type = T;
    using Point     = Array<T, 2>;
    //!@}

  public:
    // Construct with left and right values for x and y
    inline CELER_FUNCTION Interpolator(Point left, Point right);

    // Interpolate
    inline CELER_FUNCTION real_type operator()(real_type x) const;

  private:
    real_type intercept_; //!> f_y(y_l)
    real_type slope_;     //!> ratio of g(y) to g(x)
    real_type offset_;    //!> n_x(x_l)

    using XTraits_t = detail::InterpolatorTraits<XI, real_type>;
    using YTraits_t = detail::InterpolatorTraits<YI, real_type>;
};

//! Linear interpolation
template<class T>
using LinearInterpolator = Interpolator<Interp::linear, Interp::linear, T>;

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "Interpolator.i.hh"
