//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/detail/SpanUtilsImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Load a const span into a fixed-size array.
template<class T, std::size_t N>
CELER_FUNCTION Array<T, N> load_array(Span<T const, N> s)
{
    static_assert(N != dynamic_extent);
    Array<T, N> result;
    for (std::size_t i = 0; i != N; ++i)
        result[i] = s[i];
    return result;
}

//---------------------------------------------------------------------------//
//! Store a fixed-size array into a span.
template<class T, std::size_t N>
CELER_FUNCTION void store_array(Array<T, N> const& a, Span<T, N> dst)
{
    static_assert(N != dynamic_extent);
    for (std::size_t i = 0; i != N; ++i)
        dst[i] = a[i];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
