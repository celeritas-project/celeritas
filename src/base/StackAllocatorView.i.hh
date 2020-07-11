//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorView.i.hh
//---------------------------------------------------------------------------//

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
CELER_FUNCTION bool StackAllocatorView::valid() const
{
    return storage.empty() == bool(size);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
