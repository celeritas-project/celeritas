//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file corecel/math/ArrayOperators.hh
 * \brief Mathematical operators for the Array type.
 *
 * For performance reasons, avoid chaining these operators together: unroll
 * arithmetic when possible. Note that all types must be consistent (unless
 * promotion is automatically applied to scalar arguments), so you cannot
 * multiply an array of doubles with an array of ints without explicitly
 * converting first.
 */
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/cont/Array.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
#define CELER_DEFINE_ARRAY_ASSIGN(TOKEN)                                    \
    template<class T, size_type N>                                          \
    inline CELER_FUNCTION Array<T, N>& operator TOKEN(Array<T, N>& x,       \
                                                      Array<T, N> const& y) \
    {                                                                       \
        for (size_type i = 0; i != N; ++i)                                  \
        {                                                                   \
            x[i] TOKEN y[i];                                                \
        }                                                                   \
        return x;                                                           \
    }                                                                       \
                                                                            \
    template<class T, size_type N>                                          \
    inline CELER_FUNCTION Array<T, N>& operator TOKEN(Array<T, N>& x,       \
                                                      T const& y)           \
    {                                                                       \
        for (size_type i = 0; i != N; ++i)                                  \
        {                                                                   \
            x[i] TOKEN y;                                                   \
        }                                                                   \
        return x;                                                           \
    }

#define CELER_DEFINE_ARRAY_ARITHM(TOKEN)                                   \
    template<class T, size_type N>                                         \
    inline CELER_FUNCTION Array<T, N> operator TOKEN(Array<T, N> const& x, \
                                                     Array<T, N> const& y) \
    {                                                                      \
        Array<T, N> result{x};                                             \
        return (result TOKEN## = y);                                       \
    }                                                                      \
                                                                           \
    template<class T, size_type N, class T2 = std::remove_cv_t<T>>         \
    inline CELER_FUNCTION Array<T, N> operator TOKEN(Array<T, N> const& x, \
                                                     T2 const& y)          \
    {                                                                      \
        Array<T, N> result{x};                                             \
        return (result TOKEN## = y);                                       \
    }

//---------------------------------------------------------------------------//
//!@{
//! Assignment arithmetic
CELER_DEFINE_ARRAY_ASSIGN(+=)
CELER_DEFINE_ARRAY_ASSIGN(-=)
CELER_DEFINE_ARRAY_ASSIGN(*=)
CELER_DEFINE_ARRAY_ASSIGN(/=)
//!@}

//---------------------------------------------------------------------------//
//!@{
//! Arithmetic
CELER_DEFINE_ARRAY_ARITHM(+)
CELER_DEFINE_ARRAY_ARITHM(-)
CELER_DEFINE_ARRAY_ARITHM(*)
CELER_DEFINE_ARRAY_ARITHM(/)
//!@}

//! Left-multiply by scalar
template<class T, size_type N, class T2 = std::remove_cv_t<T>>
inline CELER_FUNCTION Array<T, N> operator*(T2 const& y, Array<T, N> const& x)
{
    return x * y;
}

//---------------------------------------------------------------------------//
/*!
 * Unary negation.
 */
template<class T, size_type N>
inline CELER_FUNCTION Array<T, N> operator-(Array<T, N> const& x)
{
    Array<T, N> result;
    for (size_type i = 0; i != N; ++i)
    {
        result[i] = -x[i];
    }
    return result;
}

#undef CELER_DEFINE_ARRAY_ASSIGN
#undef CELER_DEFINE_ARRAY_ARITHM
//---------------------------------------------------------------------------//
}  // namespace celeritas
