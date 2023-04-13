//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SortTracksAction.cc
//---------------------------------------------------------------------------//
#include "SortTracksAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/track/detail/TrackSortUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data
 */
void SortTracksAction::execute(CoreHostRef const& core) const
{
    detail::partition_tracks_by_status(core.states);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data
 */
void SortTracksAction::execute(
    [[maybe_unused]] CoreDeviceRef const& core) const
{
#if !CELER_USE_DEVICE
    CELER_NOT_CONFIGURED("CUDA OR HIP");
#else
    detail::partition_tracks_by_status(core.states);
#endif
}

}  // namespace celeritas
