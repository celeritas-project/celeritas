//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/SoftEqual.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "detail/SoftEqualTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Functor for floating point equality.
 *
 * \note This functor is *not commutative*: eq(a,b) will not always give the
 * same as eq(b,a).
 */
template<class RealType = ::celeritas::real_type>
class SoftEqual
{
  public:
    //!@{
    //! Type aliases
    using value_type = RealType;
    //!@}

  public:
    // Construct with default relative/absolute precision
    inline CELER_FUNCTION SoftEqual();

    // Construct with default absolute precision
    inline explicit CELER_FUNCTION SoftEqual(value_type rel);

    // Construct with both relative and absolute precision
    inline CELER_FUNCTION SoftEqual(value_type rel, value_type abs);

    //// COMPARISON ////

    // Compare two values (implicitly casting arguments)
    bool CELER_FUNCTION operator()(value_type expected,
                                   value_type actual) const;

    //// ACCESSORS ////

    //! Relative allowable error
    CELER_FUNCTION value_type rel() const { return rel_; }

    //! Absolute tolerance
    CELER_FUNCTION value_type abs() const { return abs_; }

  private:
    value_type rel_;
    value_type abs_;

    using traits_t = detail::SoftEqualTraits<value_type>;
};

//---------------------------------------------------------------------------//
/*!
 * Functor for floating point equality.
 */
template<class RealType = ::celeritas::real_type>
class SoftZero
{
  public:
    //!@{
    //! Type aliases
    using argument_type = RealType;
    using value_type = RealType;
    //!@}

  public:
    // Construct with default relative/absolute precision
    inline CELER_FUNCTION SoftZero();

    // Construct with absolute precision
    inline explicit CELER_FUNCTION SoftZero(value_type abs);

    //// COMPARISON ////

    // Compare the given value to zero
    inline CELER_FUNCTION bool operator()(value_type actual) const;

    //// ACCESSORS ////

    //! Absolute tolerance
    CELER_FUNCTION value_type abs() const { return abs_; }

  private:
    value_type abs_;

    using traits_t = detail::SoftEqualTraits<value_type>;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
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
    const value_type abs_a = std::fabs(actual);
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
//! Soft equivalence with default tolerance
template<class RealType>
inline CELER_FUNCTION bool soft_equal(RealType expected, RealType actual)
{
    return SoftEqual<RealType>()(expected, actual);
}

//---------------------------------------------------------------------------//
//! Soft equivalence with relative error
template<class RealType>
inline CELER_FUNCTION bool
soft_near(RealType expected, RealType actual, RealType rel_error)
{
    return SoftEqual<RealType>(rel_error)(expected, actual);
}

//---------------------------------------------------------------------------//
//! Soft equivalence to zero, with default tolerance
template<class RealType>
inline CELER_FUNCTION bool soft_zero(RealType actual)
{
    return SoftZero<RealType>()(actual);
}

//---------------------------------------------------------------------------//
//! Soft modulo operator
template<class RealType>
inline CELER_FUNCTION bool soft_mod(RealType dividend, RealType divisor)
{
    auto remainder = std::fmod(dividend, divisor);

    SoftEqual<RealType> seq(detail::SoftEqualTraits<RealType>::rel_prec());

    return seq(0, remainder) || seq(divisor, remainder);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
