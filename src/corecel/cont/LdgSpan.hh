//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/LdgSpan.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>

#include "corecel/data/LdgRefWrapper.hh"

#include "Span.hh"

#include "detail/LdgIterator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Alias for a Span iterating over device const values read using __ldg
template<class T, std::size_t Extent = dynamic_extent>
using LdgSpan = Span<LdgRefWrapper<T>, Extent>;

//---------------------------------------------------------------------------//
/*!
 * Construct an array from a fixed-size span, removing LdgValue marker.
 *
 * Note: \code to_array(Span<T,N> const&) \endcode is not reused because:
 * 1. Using this overload reads input data using \c __ldg
 * 2. \code return to_array<T, N>(s) \endcode results in segfault (gcc 11.3).
 *    This might be a compiler bug because temporary lifetime should be
 *    extended until the end of the expression and we return a copy...
 */
template<class T, std::size_t N>
CELER_CONSTEXPR_FUNCTION auto to_array(Span<LdgRefWrapper<T const>, N> s)
{
    Array<std::remove_cv_t<T>, N> result{};
    for (std::size_t i = 0; i < N; ++i)
    {
        result[i] = s[i];
    }
    return result;
}

//---------------------------------------------------------------------------//
//! Cast an LdgSpan to a regular Span
template<class T, std::size_t N>
CELER_CONSTEXPR_FUNCTION Span<T const, N> make_span(LdgSpan<T const, N> cont)
{
    return {cont.data(), cont.size()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
