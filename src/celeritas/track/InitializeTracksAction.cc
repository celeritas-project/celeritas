//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/InitializeTracksAction.cc
//---------------------------------------------------------------------------//
#include "InitializeTracksAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/track/TrackInitUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data
 */
void InitializeTracksAction::execute(CoreParams const& params,
                                     StateHostRef& states) const
{
    initialize_tracks(params, states);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data
 */
void InitializeTracksAction::execute(CoreParams const& params,
                                     StateDeviceRef& states) const
{
    initialize_tracks(params, states);
}

}  // namespace celeritas
