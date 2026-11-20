//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ReprRange.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>

#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"

#include "detail/ReprContainer.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class T, std::size_t N>
auto repr(Array<T, N> const& obj)
{
    auto const* data = obj.data();
    return detail::ReprContainer<T const*>{data, data + N, N > 8};
}

//---------------------------------------------------------------------------//
template<class T, std::size_t Extent>
auto repr(Span<T, Extent> const& obj)
{
    return detail::ReprContainer{obj.begin(), obj.end(), obj.size() > 8};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
