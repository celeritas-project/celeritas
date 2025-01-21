//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * Square root of the soft equivalence tolerance for Celeritas.
 *
 * This tolerance is needed for operations where the accuracy is limited by the
 * square root of machine precision.
 *
 * \todo Move orange tolerance and related operations into corecel/math
 * alongside this, revisit ArrayUtils soft comparisons.
 */
CELER_CONSTEXPR_FUNCTION real_type sqrt_tol()
{
    return detail::SoftEqualTraits<real_type>::sqrt_prec();
}

//---------------------------------------------------------------------------//
/*!
 * Functor for noninfinite floating point equality.
 *
 * This function-like class considers an \em absolute tolerance for values near
 * zero, and a \em relative tolerance for values far from zero. It correctly
 * returns "false" if either value being compared is NaN.  The call operator is
 * \em commutative: \c eq(a,b) should always give the same as \c eq(b,a).
 *
 * The actual comparison is: \f[
 |a - b| < \max(\epsilon_r \max(|a|, |b|), \epsilon_a)
 \f]
 *
 * \note The edge case where both values are infinite (with the same sign)
 * returns \em false for equality, which could be considered reasonable because
 * relative error is meaningless. To explicitly allow infinities to compare
 * equal, you must test separately, e.g., `a == b || soft_eq(a, b)` using \c
 * celeritas::EqualOr.
 */
template<class RealType = ::celeritas::real_type>
class SoftEqual
{
  public:
    //!@{
    //! \name Type aliases
    using value_type = RealType;
    //!@}

  public:
    // Construct with default relative/absolute precision
    CELER_CONSTEXPR_FUNCTION SoftEqual();

    // Construct with default absolute precision
    explicit CELER_FUNCTION SoftEqual(value_type rel);

    // Construct with both relative and absolute precision
    CELER_FUNCTION SoftEqual(value_type rel, value_type abs);

    //// COMPARISON ////

    // Compare two values (implicitly casting arguments)
    bool CELER_FUNCTION operator()(value_type a, value_type b) const;

    //// ACCESSORS ////

    //! Relative allowable error
    CELER_CONSTEXPR_FUNCTION value_type rel() const { return rel_; }

    //! Absolute tolerance
    CELER_CONSTEXPR_FUNCTION value_type abs() const { return abs_; }

  private:
    value_type rel_;
    value_type abs_;

    using SETraits = detail::SoftEqualTraits<value_type>;
};

//---------------------------------------------------------------------------//
/*!
 * Compare for equality before checking with the given functor.
 *
 * This CRTP class allows \c SoftEqual to work for infinities.
 */
template<class F>
class EqualOr : public F
{
  public:
    //! Forward arguments to parent class
    template<class... C>
    CELER_FUNCTION EqualOr(C&&... args) : F{std::forward<C>(args)...}
    {
    }

    //! Forward arguments to comparison operator after comparing
    template<class T, class U>
    bool CELER_FUNCTION operator()(T a, U b) const
    {
        return a == b || static_cast<F const&>(*this)(a, b);
    }
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
    //! \name Type aliases
    using argument_type = RealType;
    using value_type = RealType;
    //!@}

  public:
    // Construct with default relative/absolute precision
    CELER_CONSTEXPR_FUNCTION SoftZero();

    // Construct with absolute precision
    explicit CELER_FUNCTION SoftZero(value_type abs);

    //// COMPARISON ////

    // Compare the given value to zero
    inline CELER_FUNCTION bool operator()(value_type actual) const;

    //// ACCESSORS ////

    //! Absolute tolerance
    CELER_CONSTEXPR_FUNCTION value_type abs() const { return abs_; }

  private:
    value_type abs_;

    using SETraits = detail::SoftEqualTraits<value_type>;
};

//---------------------------------------------------------------------------//
// TEMPLATE DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class T>
CELER_FUNCTION SoftEqual(T) -> SoftEqual<T>;
template<class T>
CELER_FUNCTION SoftEqual(T, T) -> SoftEqual<T>;
template<class F>
CELER_FUNCTION EqualOr(F&&) -> EqualOr<F>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with default relative/absolute precision.
 */
template<class RealType>
CELER_CONSTEXPR_FUNCTION SoftEqual<RealType>::SoftEqual()
    : rel_{SETraits::rel_prec()}, abs_{SETraits::abs_thresh()}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with scaled absolute precision.
 */
template<class RealType>
CELER_FUNCTION SoftEqual<RealType>::SoftEqual(value_type rel)
    : SoftEqual(rel, rel * (SETraits::abs_thresh() / SETraits::rel_prec()))
{
    CELER_EXPECT(rel > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with both relative and absolute precision.
 *
 * \param rel tolerance of relative error (default 1.0e-12 for doubles)
 *
 * \param abs threshold for absolute error when comparing small quantities
 *           (default 1.0e-14 for doubles)
 */
template<class RealType>
CELER_FUNCTION SoftEqual<RealType>::SoftEqual(value_type rel, value_type abs)
    : rel_{rel}, abs_{abs}
{
    CELER_EXPECT(rel > 0);
    CELER_EXPECT(abs > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Compare two values, implicitly casting arguments.
 */
template<class RealType>
CELER_FUNCTION bool
SoftEqual<RealType>::operator()(value_type a, value_type b) const
{
    real_type rel = rel_ * std::fmax(std::fabs(a), std::fabs(b));
    return std::fabs(a - b) < std::fmax(abs_, rel);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with default relative/absolute precision.
 */
template<class RealType>
CELER_CONSTEXPR_FUNCTION SoftZero<RealType>::SoftZero()
    : abs_(SETraits::abs_thresh())
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with specified precision.
 *
 * \param abs threshold for absolute error (default 1.0e-14 for doubles)
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
