//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SoftEqual.hh
//---------------------------------------------------------------------------//
#pragma once

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
 *
 * \param rel tolerance of relative error (default 1.0e-12 for doubles)
 *
 * \param abs threshold for absolute error when comparing to zero
 *           (default 1.0e-14 for doubles)
 */
template<typename T1 = real_type, typename T2 = T1>
class SoftEqual
{
  public:
    //!@{
    //! Type aliases
    using first_argument_type  = T1;
    using second_argument_type = T2;
    using value_type =
        typename detail::SoftPrecisionType<first_argument_type,
                                           second_argument_type>::type;
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
 *
 * \param abs threshold for absolute error when comparing to zero
 *           (default 1.0e-14 for doubles)
 */
template<typename T>
class SoftZero
{
  public:
    //!@{
    //! Type aliases
    using argument_type = T;
    using value_type    = T;
    using traits_t      = detail::SoftEqualTraits<value_type>;
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
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "SoftEqual.i.hh"
