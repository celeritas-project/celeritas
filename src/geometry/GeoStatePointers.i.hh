//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoStatePointers.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Check whether the state is consistently assigned.
 *
 * This is called as part of the bool operator, which should be checked as part
 * of an assertion immediately before launching a kernel and when returning a
 * state.
 */
CELER_FUNCTION bool GeoStatePointers::valid() const
{
    // clang-format off
    return    bool(size) == bool(vgmaxdepth)
           && bool(size) == bool(vgstate)
           && bool(size) == bool(vgnext)
           && bool(size) == bool(pos)
           && bool(size) == bool(dir)
           && bool(size) == bool(next_step);
    // clang-format on
}

//---------------------------------------------------------------------------//
} // namespace celeritas
