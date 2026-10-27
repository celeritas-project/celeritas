//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/HandedBackIds.cc
//---------------------------------------------------------------------------//
#include "HandedBackIds.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Mark a track ID as handed back in the given event.
 */
void HandedBackIds::insert(int event_id, int track_id)
{
    CELER_EXPECT(event_id >= 0);
    CELER_EXPECT(track_id > 0);

    this->update_event(event_id);
    ids_.insert(track_id);
}

//---------------------------------------------------------------------------//
/*!
 * Unmark a track ID, returning whether it was handed back in the event.
 */
bool HandedBackIds::erase(int event_id, int track_id)
{
    CELER_EXPECT(event_id >= 0);

    this->update_event(event_id);
    return ids_.erase(track_id) > 0;
}

//---------------------------------------------------------------------------//
/*!
 * Discard the IDs of a previous event.
 *
 * Remaining IDs belong to handed-back tracks that the Celeritas tracking
 * manager never saw again: they may have been deleted by Geant4, tracked by
 * the standard Geant4 tracking manager (if Celeritas does not offload their
 * particle type), or postponed to the next event.
 */
void HandedBackIds::update_event(int event_id)
{
    if (event_id == event_id_)
    {
        return;
    }
    if (!ids_.empty())
    {
        CELER_LOG_LOCAL(debug)
            << ids_.size() << " handed-back tracks from event " << event_id_
            << " were not tracked by the Celeritas tracking manager";
        ids_.clear();
    }
    event_id_ = event_id;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
