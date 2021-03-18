//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SoftEqual.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "detail/SoftEqualTraits.hh"
#include "Macros.hh"
#include "Types.hh"

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
    using value_type    = RealType;
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

} // namespace celeritas

#include "SoftEqual.i.hh"
