//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromSecondariesAction.cc
//---------------------------------------------------------------------------//
#include "ExtendFromSecondariesAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/track/TrackInitUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data
 */
void ExtendFromSecondariesAction::execute(CoreHostRef const& core) const
{
    initialize_tracks(core);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data
 */
void ExtendFromSecondariesAction::execute(
    [[maybe_unused]] CoreDeviceRef const& core) const
{
#if !CELER_USE_DEVICE
    CELER_NOT_CONFIGURED("CUDA OR HIP");
#else
    initialize_tracks(core);
#endif
}

}  // namespace celeritas
