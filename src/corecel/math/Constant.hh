//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Constant.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Full-precision floating point constant with automatic precision demotion.
 *
 * We want two behaviors from constants in Celeritas:
 * 1. They don't accidentally promote runtime arithmetic from single to double
 *    precision when compiling at a lower precision. This incurs a substantial
 *    performance penalty on GPU.
 * 2. We can use their full double-precision values when we need to: either in
 *    templated code or when interacting with other libraries. (For example,
 *    float(pi) > pi which can lead to errors in Geant4.)
 *
 * This class stores a full-precision (double) value as its "real type" and
 * defines explicit conversion operators that allow it to automatically convert
 * to a lower-precision or real-precision type.
 *
 * Operations with a floating point value returns a value of that precision
 * (performed at that precision level); operations with integers return a
 * full-precision Constant; and operations with Constants return a Constant.
 */
class Constant
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = double;
    //!@}

  public:
    //! Explicitly construct from a full-precision value
    explicit CELER_CONSTEXPR_FUNCTION Constant(real_type v) : value_{v} {}

    //! Access the value explicitly
    CELER_CONSTEXPR_FUNCTION real_type value() const { return value_; }

    //! Explicit conversion of stored value
    explicit CELER_CONSTEXPR_FUNCTION operator float() const { return value_; }
    //! Explicit access to stored value
    explicit CELER_CONSTEXPR_FUNCTION operator double() const
    {
        return value_;
    }

  private:
    real_type value_;
};

//! Unary negation
CELER_CONSTEXPR_FUNCTION Constant operator-(Constant lhs) noexcept
{
    return Constant{-lhs.value()};
}

//---------------------------------------------------------------------------//
//! \cond
#define CELER_DEFINE_CONSTANT_CMP(TOKEN)                                          \
    template<class T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true> \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(Constant lhs, T rhs) noexcept    \
    {                                                                             \
        return static_cast<T>(lhs.value()) TOKEN rhs;                             \
    }                                                                             \
    template<class T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true> \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(T lhs, Constant rhs) noexcept    \
    {                                                                             \
        return lhs TOKEN static_cast<T>(rhs.value());                             \
    }                                                                             \
    template<class T, std::enable_if_t<std::is_integral_v<T>, bool> = true>       \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(Constant lhs, T rhs) noexcept    \
    {                                                                             \
        return lhs.value() TOKEN static_cast<Constant::real_type>(rhs);           \
    }                                                                             \
    template<class T, std::enable_if_t<std::is_integral_v<T>, bool> = true>       \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(T lhs, Constant rhs) noexcept    \
    {                                                                             \
        return static_cast<Constant::real_type>(lhs) TOKEN rhs.value();           \
    }                                                                             \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(Constant lhs,                    \
                                                 Constant rhs) noexcept           \
    {                                                                             \
        return lhs.value() TOKEN rhs.value();                                     \
    }

//!@{
//! Comparison for Constant
CELER_DEFINE_CONSTANT_CMP(==)
CELER_DEFINE_CONSTANT_CMP(!=)
CELER_DEFINE_CONSTANT_CMP(<)
CELER_DEFINE_CONSTANT_CMP(>)
CELER_DEFINE_CONSTANT_CMP(<=)
CELER_DEFINE_CONSTANT_CMP(>=)
//!@}

#undef CELER_DEFINE_CONSTANT_CMP

//!@{
//! Arithmetic for Constant
#define CELER_DEFINE_CONSTANT_OP(TOKEN)                                           \
    template<class T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true> \
    CELER_CONSTEXPR_FUNCTION T operator TOKEN(Constant lhs, T rhs) noexcept       \
    {                                                                             \
        return static_cast<T>(lhs.value()) TOKEN rhs;                             \
    }                                                                             \
    template<class T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true> \
    CELER_CONSTEXPR_FUNCTION T operator TOKEN(T lhs, Constant rhs) noexcept       \
    {                                                                             \
        return lhs TOKEN static_cast<T>(rhs.value());                             \
    }                                                                             \
    template<class T, std::enable_if_t<std::is_integral_v<T>, bool> = true>       \
    CELER_CONSTEXPR_FUNCTION Constant operator TOKEN(Constant lhs,                \
                                                     T rhs) noexcept              \
    {                                                                             \
        return Constant{lhs.value() TOKEN rhs};                                   \
    }                                                                             \
    template<class T, std::enable_if_t<std::is_integral_v<T>, bool> = true>       \
    CELER_CONSTEXPR_FUNCTION Constant operator TOKEN(T lhs,                       \
                                                     Constant rhs) noexcept       \
    {                                                                             \
        return Constant{lhs TOKEN rhs.value()};                                   \
    }                                                                             \
    CELER_CONSTEXPR_FUNCTION Constant operator TOKEN(Constant lhs,                \
                                                     Constant rhs) noexcept       \
    {                                                                             \
        return Constant{lhs.value() TOKEN rhs.value()};                           \
    }

CELER_DEFINE_CONSTANT_OP(*)
CELER_DEFINE_CONSTANT_OP(/)
CELER_DEFINE_CONSTANT_OP(+)
CELER_DEFINE_CONSTANT_OP(-)
//!@!}

#undef CELER_DEFINE_CONSTANT_OP

//!@{
//! In-place arithmetic for Constant
#define CELER_DEFINE_CONSTANT_OP(TOKEN)                                           \
    template<class T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true> \
    CELER_CONSTEXPR_FUNCTION T& operator TOKEN(T & lhs, Constant rhs) noexcept    \
    {                                                                             \
        return lhs TOKEN static_cast<T>(rhs.value());                             \
    }

CELER_DEFINE_CONSTANT_OP(*=)
CELER_DEFINE_CONSTANT_OP(/=)
CELER_DEFINE_CONSTANT_OP(+=)
CELER_DEFINE_CONSTANT_OP(-=)
//!@!}

#undef CELER_DEFINE_CONSTANT_OP

//! \endcond
//---------------------------------------------------------------------------//
}  // namespace celeritas
