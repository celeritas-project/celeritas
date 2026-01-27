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
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    ScoringTrackView ...;
   \endcode
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
 */
CELER_FUNCTION
ScoringTrackView::ScoringTrackView(StateRef const& state, TrackSlotId tid)
    : state_(state), track_(tid)
{
    CELER_EXPECT(tid);
}

//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION void ScoringTrackView::clear_hit()
{
    this->hit().detector = {};
}

//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION void ScoringTrackView::score_hit(DetectorHit hit)
{
    CELER_EXPECT(hit.detector);
    this->hit() = std::move(hit);
}

//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION DetectorHit& ScoringTrackView::hit()
{
    CELER_EXPECT(track_ < state_.size());
    return state_.all_track_hits[track_];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
