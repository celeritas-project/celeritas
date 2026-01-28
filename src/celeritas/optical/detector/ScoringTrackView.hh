//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/ScoringTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"

#include "DetectorData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Track view into the corresponding hit buffer.
 *
 * For a given track ID, this view accesses the corresponding hit data in the
 * detector state buffer. For tracks in detectors, the \c score_hit function
 * may be used to populate the track's hit data. Otherwise, the track's hit
 * data should be cleared at the end of every step with \c clear_hit.
 */
class ScoringTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using StateRef = NativeRef<DetectorStateData>;
    //!@}

  public:
    // Construct from local data
    inline CELER_FUNCTION ScoringTrackView(StateRef const&, TrackSlotId);

    // Clear hit data for this track
    inline CELER_FUNCTION void clear_hit();

    // Score hit for this track
    inline CELER_FUNCTION void score_hit(DetectorHit);

    // Get hit data associated with this track
    inline CELER_FUNCTION DetectorHit& hit();

  private:
    StateRef const& state_;
    TrackSlotId track_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from detector state buffer and track ID.
 */
CELER_FUNCTION
ScoringTrackView::ScoringTrackView(StateRef const& state, TrackSlotId tid)
    : state_(state), track_(tid)
{
    CELER_EXPECT(tid);
}

//---------------------------------------------------------------------------//
/*!
 * Clear hit data for this track.
 *
 * Marks the hit with an invalid detector ID.
 */
CELER_FUNCTION void ScoringTrackView::clear_hit()
{
    this->hit().detector = {};
}

//---------------------------------------------------------------------------//
/*!
 * Set the hit data for this track.
 *
 * Should have a valid detector ID to indicate it is a valid hit.
 */
CELER_FUNCTION void ScoringTrackView::score_hit(DetectorHit hit)
{
    CELER_EXPECT(hit.detector);
    this->hit() = std::move(hit);
}

//---------------------------------------------------------------------------//
/*!
 * Access the hit data associated with this track.
 */
CELER_FUNCTION DetectorHit& ScoringTrackView::hit()
{
    CELER_EXPECT(track_ < state_.size());
    return state_.all_track_hits[track_];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
