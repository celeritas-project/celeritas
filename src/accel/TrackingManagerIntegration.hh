//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerIntegration.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

class G4Run;

namespace celeritas
{
//---------------------------------------------------------------------------//
struct SetupOptions;
class TrackingManager;
class TrackingManagerIntegration;

//---------------------------------------------------------------------------//
/*!
 * Simple interface for G4VTrackingManager-based integration.
 *
 * This singleton integrates both thread-local and global data with the user
 * application. To use this class in your Geant4 application to offload tracks
 * to Celeritas:
 *
 * - Set up the \c Options before calling \c G4RunManager::Initialize: usually
 *   in \c main .
 * - Call \c Build and \c BuildForMaster from the corresponding \c
 *   G4UserActionInitialization::Build functions
 * - Call \c BeginOfRunAction and \c EndOfRunAction from the corresponding \c
 * G4UserRunAction
 *
 * The \c CELER_DISABLE environment variable, if set and non-empty, will
 * disable offloading so that Celeritas will not be built nor kill tracks.
 *
 * The method names correspond to methods in Geant4 User Actions and \em must
 * be called from all threads, both worker and master.
 *
 * \todo Provide default minimal action initialization classes for user
 */
class TrackingManagerIntegration
{
  public:
    // Access the public-facing integration singleton
    static TrackingManagerIntegration& Instance();

    // Edit options before starting the run
    SetupOptions& Options();

    // Initialize during ActionInitialization on non-worker thread
    void BuildForMaster();

    // Initialize during ActionInitialization
    void Build();

    // Start the run
    void BeginOfRunAction(G4Run const* run);

    // End the run
    void EndOfRunAction(G4Run const* run);

  private:
    // Tracking manager can only be created privately
    TrackingManagerIntegration();
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
