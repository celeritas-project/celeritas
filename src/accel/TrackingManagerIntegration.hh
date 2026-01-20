//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerIntegration.hh
//---------------------------------------------------------------------------//
#pragma once

#include "IntegrationBase.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simple interface for G4VTrackingManager-based integration.
 *
 * This singleton integrates both thread-local and global data with the user
 * application. To use this class in your Geant4 application to offload tracks
 * to Celeritas:
 *
 * - Use the \c TrackingManagerConstructor class to add the Celeritas tracking
 *   manager to your physics list.
 * - Use \c SetOptions to set up options before \c G4RunManager::Initialize:
 *   usually in \c main for simple applications.
 * - Call \c BeginOfRunAction and \c EndOfRunAction from \c UserRunAction
 *
 * See further documentation in \c celeritas::IntegrationBase.
 */
class TrackingManagerIntegration final : public IntegrationBase
{
  public:
    // Access the public-facing integration singleton
    static TrackingManagerIntegration& Instance();

  private:
    // Tracking manager can only be created privately
    TrackingManagerIntegration();

    // Verify tracking manager setup
    void verify_local_setup() final;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
