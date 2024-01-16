//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/HyperslabIndexerImpl.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace detail
{
// Utility function for calculating the size of hyperslab data
template<celeritas::size_type N>
inline CELER_FUNCTION celeritas::size_type
hyperslab_size(celeritas::Array<celeritas::size_type, N> const& dims)
{
    celeritas::size_type size = 1;
    for (auto const dim : dims)
    {
        size *= dim;
    }
    return size;
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
