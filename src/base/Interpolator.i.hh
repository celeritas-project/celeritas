//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Interpolator.i.hh
//---------------------------------------------------------------------------//

#include "Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// INTERPOLATOR
//---------------------------------------------------------------------------//
/*!
 * \brief Construct with left and right values for x and y
 */
template<Interp IX, Interp IY, typename T>
CELER_FUNCTION Interpolator<IX, IY, T>::Interpolator(Point left, Point right)
{
    enum
    {
        X = 0,
        Y = 1
    };

    REQUIRE(left[X] != right[X]);
    REQUIRE(XTraits_t::valid_domain(left[X]));
    REQUIRE(XTraits_t::valid_domain(right[X]));
    REQUIRE(YTraits_t::valid_domain(left[Y]));
    REQUIRE(YTraits_t::valid_domain(right[Y]));

    intercept_ = YTraits_t::transform(left[Y]);
    slope_     = (YTraits_t::add_transformed(
                  YTraits_t::negate_transformed(left[Y]), right[Y])
              / XTraits_t::add_transformed(
                  XTraits_t::negate_transformed(left[X]), right[X]));
    offset_    = XTraits_t::negate_transformed(left[X]);
    ENSURE(!std::isnan(intercept_) && !std::isnan(slope_)
           && !std::isnan(offset_));
}

//---------------------------------------------------------------------------//
/*!
 * \brief Interpolate linearly on the transformed type.
 */
template<Interp IX, Interp IY, typename T>
CELER_FUNCTION auto Interpolator<IX, IY, T>::operator()(real_type x) const
    -> real_type
{
    REQUIRE(XTraits_t::valid_domain(x));
    real_type result = YTraits_t::transform_inv(
        intercept_ + slope_ * (XTraits_t::add_transformed(offset_, x)));

    ENSURE(!std::isnan(result));
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
