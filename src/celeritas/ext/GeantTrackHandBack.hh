//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantTrackHandBack.hh
//! \sa test/accel/TrackingManagerIntegration.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <unordered_set>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"

class G4Track;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return reconstructed tracks to Geant4 and track them on CPU.
 *
 * Handing a track back transfers its ownership, including its user
 * information, to the current Geant4 event by pushing it onto the track stack
 * with \c G4EventManager::StackTracks , which invokes the user stacking
 * action. Reconstructed tracks already have their Geant4 track ID (see \c
 * GeantTrackReconstruction ), so Geant4 does not assign a new one.
 *
 * Since Celeritas transports some particle types with a custom tracking
 * manager, Geant4 would offload a handed-back track again as soon as it pops
 * it from the stack. The tracking manager must therefore call \c process
 * first: if the track was handed back, it is tracked by the standard Geant4
 * tracking manager, mirroring \c G4EventManager::DoProcessing :
 * - its secondaries are stacked with new track IDs,
 * - its trajectory, if any, is stored in the event,
 * - if it is suspended or postponed, it is pushed back to the stack \em
 *   without being marked as handed back, so that the next time it is popped
 *   it is offloaded to Celeritas again with the same ID, and otherwise
 * - the track is deleted.
 *
 * \warning This class is thread-local: it must be used on the worker thread
 * that owns the tracks, while an event is being processed.
 */
class GeantTrackHandBack
{
  public:
    //!@{
    //! \name Type aliases
    using UPTrack = std::unique_ptr<G4Track>;
    //!@}

  public:
    // Construct with no tracks handed back
    GeantTrackHandBack() = default;

    // Warn about tracks still pending in the Geant4 stack
    ~GeantTrackHandBack();
    CELER_DELETE_COPY_MOVE(GeantTrackHandBack);

    // Return ownership of a reconstructed track to the current Geant4 event
    void operator()(UPTrack track);

    // Track a handed-back track on CPU, returning false if not handed back
    [[nodiscard]] bool process(G4Track* track);

    //! Number of handed-back tracks not yet tracked by Geant4
    size_type num_pending() const { return handed_back_.size(); }

    //! Number of tracks handed back since construction
    size_type num_handed_back() const { return num_handed_back_; }

  private:
    std::unordered_set<G4Track const*> handed_back_;
    size_type num_handed_back_{0};

    // Push a track to the Geant4 stack, returning false if it was killed
    static bool stack(G4Track* track);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
