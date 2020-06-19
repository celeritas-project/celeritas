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
 * Construct with default relative/absolute precision
 */
template<typename T1, typename T2>
SoftEqual<T1, T2>::SoftEqual()
    : SoftEqual(traits_t::rel_prec(), traits_t::abs_thresh())
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with scaled absolute precision
 */
template<typename T1, typename T2>
SoftEqual<T1, T2>::SoftEqual(value_type rel)
    : SoftEqual(rel, rel * (traits_t::abs_thresh() / traits_t::rel_prec()))
{
    REQUIRE(rel > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with both relative and absolute precision
 */
template<typename T1, typename T2>
SoftEqual<T1, T2>::SoftEqual(value_type rel, value_type abs)
    : d_rel(rel), d_abs(abs)
{
    REQUIRE(rel > 0);
    REQUIRE(abs > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Compare two values (implicitly casting arguments)
 *
 * \param expected scalar floating point reference to which value is compared
 * \param actual   scalar floating point value
 */
template<typename T1, typename T2>
bool SoftEqual<T1, T2>::operator()(value_type expected, value_type actual) const
{
    value_type abs_e = std::fabs(expected);

    // Typical case: relative error comparison to reference
    if (std::fabs(actual - expected) < d_rel * abs_e)
    {
        return true;
    }

    value_type eps_abs = d_abs;
    value_type abs_a   = std::fabs(actual);
    // If one is within the absolute threshold of zero, and the other within
    // relative of zero, they're equal
    if ((abs_e < eps_abs) && (abs_a < d_rel))
    {
        return true;
    }
    if ((abs_a < eps_abs) && (abs_e < d_rel))
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
 * Construct with default relative/absolute precision
 */
template<typename T>
SoftZero<T>::SoftZero() : SoftZero(traits_t::abs_thresh())
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with default absolute precision
 */
template<typename T>
SoftZero<T>::SoftZero(value_type abs) : d_abs(abs)
{
    REQUIRE(abs > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Compare value against zero
 *
 * \param actual   scalar floating point value
 */
template<typename T>
bool SoftZero<T>::operator()(value_type actual) const
{
    // Return whether the absolute value is within tolerance
    return std::fabs(actual) < d_abs;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
