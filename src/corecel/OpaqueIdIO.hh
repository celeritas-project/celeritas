//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/OpaqueIdIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>

#include "OpaqueId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Output an opaque ID's value or a placeholder if unavailable.
 */
template<class V, class S>
std::ostream& operator<<(std::ostream& os, OpaqueId<V, S> const& v)
{
    if (v)
    {
        os << v.unchecked_get();
    }
    else
    {
        os << "<invalid>";
    }
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
