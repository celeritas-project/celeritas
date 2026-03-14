//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/Types.cc
//---------------------------------------------------------------------------//
#include "Types.hh"

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a geometry track state.
 */
char const* to_cstring(GeoStatus value)
{
    switch (value)
    {
        case GeoStatus::error:
            return "error";
        case GeoStatus::invalid:
            return "invalid";
        case GeoStatus::interior:
            return "interior";
        case GeoStatus::boundary_out:
            return "boundary_out";
        case GeoStatus::boundary_inc:
            return "boundary_inc";
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
