//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/UserActionIntegration.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/sys/Stopwatch.hh"

#include "IntegrationBase.hh"

class G4Event;
class G4Track;

namespace celeritas
{
//---------------------------------------------------------------------------//

struct SetupOptions;

//---------------------------------------------------------------------------//
/*!
 * Simple interface for G4VUserTrackingAction-based integration.
 *
 * This singleton integrates both thread-local and global data with the user
 * application. To use this class in your Geant4 application to offload tracks
 * to Celeritas:
 *
 * - Use \c SetOptions to set Celeritas configuration before calling \c
 *   G4RunManager::BeamOn
 * - Call \c BeginOfRunAction and \c EndOfRunAction (in \c IntegrationBase)
 *   from \c UserRunAction
 * - Call \c BeginOfEvent and  \c EndOfEvent from \c UserEventAction
 * - Call \c PreUserTrackingAction from your \c UserTrackingAction
 *
 * The method names correspond to methods in Geant4 User Actions and \em must
 * be called from all threads, both worker and master.
 *
 * See further documentation in \c celeritas::IntegrationBase.
 *
 * \note Prefer to use \c celeritas::TrackingManagerIntegration instead of this
 * class, unless you need support for Geant4 earlier than 11.0.
 */
class UserActionIntegration final : public IntegrationBase
{
  public:
    // Access the singleton
    static UserActionIntegration& Instance();

    // Send Celeritas the event ID
    void BeginOfEventAction(G4Event const* event);

    // Send tracks to Celeritas if applicable and "StopAndKill" if so
    void PreUserTrackingAction(G4Track* track);

    // Flush offloaded tracks from Celeritas
    void EndOfEventAction(G4Event const* event);

  private:
    // Only allow the singleton to construct
    UserActionIntegration();

    Stopwatch get_event_time_;

    // Verify setup after initialization (called if offload is enabled)
    void verify_local_setup() final;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
