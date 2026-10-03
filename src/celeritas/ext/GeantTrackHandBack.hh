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
#include <vector>

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
 * Handed-back tracks are \em deferred until \c flush is called, which stacks
 * them all in the order of \c detail::GeantTrackOrder . Stacking them as soon
 * as Celeritas returns them would insert them at an arbitrary point of the
 * Geant4 track stack, and thus of the Geant4 random number sequence, since
 * when an asynchronous step completes depends on timing. Flushing when
 * Celeritas has transported all offloaded tracks (i.e., from \c
 * G4VTrackingManager::FlushEvent ) instead makes both the set of handed-back
 * tracks and the point where they are stacked independent of the execution
 * schedule.
 *
 * \note Geant4 stops processing an event if its urgent stack is empty after
 * flushing the tracking managers, so a handed-back track that the user
 * stacking action sends to a waiting stack is \em not tracked in the current
 * event. A warning is printed if that happens.
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
 *   it is offloaded to Celeritas again with the same ID (with the Geant4
 *   geometry backend, a track that stopped on a geometry boundary is first
 *   moved slightly into the next volume so that Celeritas locates it
 *   unambiguously), and otherwise
 * - the track is deleted.
 *
 * Handing back tracks requires Geant4 11.0 or higher.
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

    // Warn about tracks not flushed or still pending in the Geant4 stack
    ~GeantTrackHandBack();
    CELER_DELETE_COPY_MOVE(GeantTrackHandBack);

    // Defer returning a reconstructed track to the current Geant4 event
    void operator()(UPTrack track);

    // Return deferred tracks to the current Geant4 event in a fixed order
    void flush();

    // Track a handed-back track on CPU, returning false if not handed back
    [[nodiscard]] bool process(G4Track* track);

    //! Number of tracks waiting to be flushed
    size_type num_deferred() const { return deferred_.size(); }

    //! Number of handed-back tracks not yet tracked by Geant4
    size_type num_pending() const { return handed_back_.size(); }

    //! Number of tracks handed back since construction
    size_type num_handed_back() const { return num_handed_back_; }

  private:
    enum class Stacked
    {
        killed,
        urgent,
        other
    };

    std::vector<UPTrack> deferred_;
    std::unordered_set<G4Track const*> handed_back_;
    size_type num_handed_back_{0};
    bool warned_not_urgent_{false};

    // Push a track to the Geant4 stack
    static Stacked stack(G4Track* track);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
