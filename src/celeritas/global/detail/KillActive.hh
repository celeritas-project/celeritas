//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/KillActive.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreParams;
template<MemSpace M>
class CoreState;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Mark active tracks as "errored".
 */
struct KillActiveExecutor
{
    inline CELER_FUNCTION void operator()(celeritas::CoreTrackView& track);
};

//---------------------------------------------------------------------------//

void kill_active(CoreParams const& params, CoreState<MemSpace::host>& state);
void kill_active(CoreParams const& params, CoreState<MemSpace::device>& state);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
CELER_FUNCTION void
KillActiveExecutor::operator()(celeritas::CoreTrackView& track)
{
    if (track.make_sim_view().status() != TrackStatus::inactive)
    {
        track.apply_errored();
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
