//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Constant.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"

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
    explicit CELER_CEF Constant(real_type v) : value_{v} {}

    //! Access the value explicitly
    CELER_CEF real_type value() const { return value_; }

    //! Explicit conversion of stored value
    explicit CELER_CEF operator float() const { return value_; }
    //! Explicit access to stored value
    explicit CELER_CEF operator double() const { return value_; }

    //// FRIENDLY OPERATORS ////

    //! Unary negation
    CELER_CEF friend Constant operator-(Constant lhs) noexcept
    {
        return Constant{-lhs.value()};
    }

  private:
    template<class T>
    using EnableIfFloating
        = std::enable_if_t<std::is_floating_point_v<T>, bool>;
    template<class T>
    using EnableIfIntegral = std::enable_if_t<std::is_integral_v<T>, bool>;

  public:
//! \cond (CELERITAS_DOC_DEV)
//!@{
//! Comparison
#define CELER_DEFINE_CONSTANT_CMP(TOKEN)                                      \
    template<class T, EnableIfFloating<T> = true>                             \
    CELER_CEF friend bool operator TOKEN(Constant lhs, T rhs) noexcept        \
    {                                                                         \
        return static_cast<T>(lhs.value()) TOKEN rhs;                         \
    }                                                                         \
    template<class T, EnableIfFloating<T> = true>                             \
    CELER_CEF friend bool operator TOKEN(T lhs, Constant rhs) noexcept        \
    {                                                                         \
        return lhs TOKEN static_cast<T>(rhs.value());                         \
    }                                                                         \
                                                                              \
    template<class T, EnableIfIntegral<T> = true>                             \
    CELER_CEF friend bool operator TOKEN(Constant lhs, T rhs) noexcept        \
    {                                                                         \
        return lhs.value() TOKEN static_cast<Constant::real_type>(rhs);       \
    }                                                                         \
    template<class T, EnableIfIntegral<T> = true>                             \
    CELER_CEF friend bool operator TOKEN(T lhs, Constant rhs) noexcept        \
    {                                                                         \
        return static_cast<Constant::real_type>(lhs) TOKEN rhs.value();       \
    }                                                                         \
                                                                              \
    CELER_CEF friend bool operator TOKEN(Constant lhs, Constant rhs) noexcept \
    {                                                                         \
        return lhs.value() TOKEN rhs.value();                                 \
    }

    CELER_DEFINE_CONSTANT_CMP(==)
    CELER_DEFINE_CONSTANT_CMP(!=)
    CELER_DEFINE_CONSTANT_CMP(<)
    CELER_DEFINE_CONSTANT_CMP(>)
    CELER_DEFINE_CONSTANT_CMP(<=)
    CELER_DEFINE_CONSTANT_CMP(>=)
    //!@}

#undef CELER_DEFINE_CONSTANT_CMP

//!@{
//! Arithmetic
#define CELER_DEFINE_CONSTANT_OP(TOKEN)                                    \
    template<class T, EnableIfFloating<T> = true>                          \
    CELER_CEF friend T operator TOKEN(Constant lhs, T rhs) noexcept        \
    {                                                                      \
        return static_cast<T>(lhs.value()) TOKEN rhs;                      \
    }                                                                      \
    template<class T, EnableIfFloating<T> = true>                          \
    CELER_CEF friend T operator TOKEN(T lhs, Constant rhs) noexcept        \
    {                                                                      \
        return lhs TOKEN static_cast<T>(rhs.value());                      \
    }                                                                      \
                                                                           \
    template<class T, EnableIfIntegral<T> = true>                          \
    CELER_CEF friend Constant operator TOKEN(Constant lhs, T rhs) noexcept \
    {                                                                      \
        return Constant{lhs.value() TOKEN rhs};                            \
    }                                                                      \
    template<class T, EnableIfIntegral<T> = true>                          \
    CELER_CEF friend Constant operator TOKEN(T lhs, Constant rhs) noexcept \
    {                                                                      \
        return Constant{lhs TOKEN rhs.value()};                            \
    }                                                                      \
                                                                           \
    CELER_CEF friend Constant operator TOKEN(Constant lhs,                 \
                                             Constant rhs) noexcept        \
    {                                                                      \
        return Constant{lhs.value() TOKEN rhs.value()};                    \
    }

    CELER_DEFINE_CONSTANT_OP(*)
    CELER_DEFINE_CONSTANT_OP(/)
    CELER_DEFINE_CONSTANT_OP(+)
    CELER_DEFINE_CONSTANT_OP(-)
    //!@!}

#undef CELER_DEFINE_CONSTANT_OP

//!@{
//! In-place arithmetic
#define CELER_DEFINE_CONSTANT_OP(TOKEN)                                \
    template<class T, EnableIfFloating<T> = true>                      \
    CELER_CEF friend T& operator TOKEN(T & lhs, Constant rhs) noexcept \
    {                                                                  \
        return lhs TOKEN static_cast<T>(rhs.value());                  \
    }

    CELER_DEFINE_CONSTANT_OP(*=)
    CELER_DEFINE_CONSTANT_OP(/=)
    CELER_DEFINE_CONSTANT_OP(+=)
    CELER_DEFINE_CONSTANT_OP(-=)
    //!@!}

#undef CELER_DEFINE_CONSTANT_OP
    //! \endcond

  private:
    real_type value_;
};

namespace literals
{
//---------------------------------------------------------------------------//
CELER_CONSTEXPR_FUNCTION Constant operator""_C(long double value)
{
    return Constant{static_cast<Constant::real_type>(value)};
}

CELER_CONSTEXPR_FUNCTION Constant operator""_C(unsigned long long int value)
{
    return Constant{static_cast<Constant::real_type>(value)};
}

//---------------------------------------------------------------------------//
}  // namespace literals
}  // namespace celeritas
