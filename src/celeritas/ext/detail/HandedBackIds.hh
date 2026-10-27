//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/HandedBackIds.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_set>

#include "corecel/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Track IDs of handed-back tracks waiting to be tracked in the current event.
 *
 * Geant4 track IDs are unique within an event, and a handed-back track is the
 * only live Geant4 track with its ID while it is on the stack. Its ID
 * therefore identifies it, even if Geant4 deletes it without tracking it
 * (e.g., when the user stacking action kills it, a stack is cleared, or the
 * event is aborted) and reuses its address for a new track.
 *
 * Track IDs repeat across events, so all IDs are discarded when the event
 * changes. A handed-back track postponed to the next event is then no longer
 * recognized, and it is offloaded again.
 */
class HandedBackIds
{
  public:
    // Mark a track ID as handed back in the given event
    void insert(int event_id, int track_id);

    // Unmark a track ID, returning whether it was handed back in the event
    [[nodiscard]] bool erase(int event_id, int track_id);

    //! Number of IDs marked in the current event
    size_type size() const { return ids_.size(); }

  private:
    int event_id_{-1};
    std::unordered_set<int> ids_;

    // Discard the IDs of a previous event
    void update_event(int event_id);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
