//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/LdgSpan.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>

#include "Span.hh"

#include "detail/LdgSpanImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Alias for a Span iterating over device const values read using \c ldg .
 *
 * This instantiates \c Span with a special wrapper class to optimize constant
 * data access in global device memory. In that case, data returned by \c
 * front, \c back, \c operator[] and \c begin / \c end iterators use value
 * semantics instead of reference.
 */
template<class T, std::size_t Extent = dynamic_extent>
using LdgSpan = Span<detail::LdgWrapper<T>, Extent>;

//---------------------------------------------------------------------------//
/*!
 * Construct an array from a fixed-size LdgSpan.
 *
 * Note: \code to_array(Span<T,N> const&) \endcode is not reused because:
 * 1. Using this overload reads input data using \c ldg
 * 2. \code return to_array<T, N>(s) \endcode results in segfault (gcc 11.3).
 *    This might be a compiler bug because temporary lifetime should be
 *    extended until the end of the expression and we return a copy...
 */
template<class T, std::size_t N>
CELER_CONSTEXPR_FUNCTION auto to_array(Span<detail::LdgWrapper<T const>, N> s)
{
    Array<std::remove_cv_t<T>, N> result{};
    for (std::size_t i = 0; i < N; ++i)
    {
        result[i] = s[i];
    }
    return result;
}

//---------------------------------------------------------------------------//
//! Convert an LdgSpan to a regular Span, \em not using \c ldg
template<class T, std::size_t N>
CELER_CONSTEXPR_FUNCTION Span<T const, N>
remove_ldg_wrapper(LdgSpan<T const, N> cont)
{
    return {cont.data(), cont.size()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
