//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SoftEqual.i.hh
//---------------------------------------------------------------------------//
#include <cmath>
#include "Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with default relative/absolute precision.
 */
template<class RealType>
CELER_FUNCTION SoftEqual<RealType>::SoftEqual()
    : SoftEqual(traits_t::rel_prec(), traits_t::abs_thresh())
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with scaled absolute precision.
 */
template<class RealType>
CELER_FUNCTION SoftEqual<RealType>::SoftEqual(value_type rel)
    : SoftEqual(rel, rel * (traits_t::abs_thresh() / traits_t::rel_prec()))
{
    CELER_EXPECT(rel > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with both relative and absolute precision.
 *
 * \param rel tolerance of relative error (default 1.0e-12 for doubles)
 *
 * \param abs threshold for absolute error when comparing to zero
 *           (default 1.0e-14 for doubles)
 */
template<class RealType>
CELER_FUNCTION SoftEqual<RealType>::SoftEqual(value_type rel, value_type abs)
    : rel_(rel), abs_(abs)
{
    CELER_EXPECT(rel > 0);
    CELER_EXPECT(abs > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Compare two values (implicitly casting arguments).
 *
 * Note that to be safe with NaN, only return \c true inside an \c if
 * conditional.
 *
 * \param expected scalar floating point reference to which value is compared
 * \param actual   scalar floating point value
 */
template<class RealType>
CELER_FUNCTION bool
SoftEqual<RealType>::operator()(value_type expected, value_type actual) const
{
    const value_type abs_e = std::fabs(expected);

    // Typical case: relative error comparison to reference
    if (std::fabs(actual - expected) < rel_ * abs_e)
    {
        return true;
    }

    const value_type eps_abs = abs_;
    const value_type abs_a   = std::fabs(actual);
    // If one is within the absolute threshold of zero, and the other within
    // relative of zero, they're equal
    if ((abs_e < eps_abs) && (abs_a < rel_))
    {
        return true;
    }
    if ((abs_a < eps_abs) && (abs_e < rel_))
    {
        return true;
    }

    // If they're both infinite and share a sign, they're equal
    if (std::isinf(expected) && std::isinf(actual)
        && std::signbit(expected) == std::signbit(actual))
    {
        return true;
    }

    return false;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with default relative/absolute precision.
 */
template<class RealType>
CELER_FUNCTION SoftZero<RealType>::SoftZero()
    : SoftZero(traits_t::abs_thresh())
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with specified precision.
 *
 * \param abs threshold for absolute error when comparing to zero
 *           (default 1.0e-14 for doubles)
 */
template<class RealType>
CELER_FUNCTION SoftZero<RealType>::SoftZero(value_type abs) : abs_(abs)
{
    CELER_EXPECT(abs > 0);
}

//---------------------------------------------------------------------------//
/*!
 * See if the value is within absolute tolerance of zero.
 *
 * \param actual scalar floating point value
 */
template<class RealType>
CELER_FUNCTION bool SoftZero<RealType>::operator()(value_type actual) const
{
    return std::fabs(actual) < abs_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
