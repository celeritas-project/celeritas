//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/FastSimulationIntegration.hh
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
 * - Add the \c FastSimulationModel to regions of interest.
 * - Use \c SetOptions to set up options before \c G4RunManager::Initialize:
 *   usually in \c main for simple applications.
 * - Call \c BeginOfRunAction and \c EndOfRunAction from \c UserRunAction
 *
 * The \c CELER_DISABLE environment variable, if set and non-empty, will
 * disable offloading so that Celeritas will not be built nor kill tracks.
 *
 * The method names correspond to methods in Geant4 User Actions and \em must
 * be called from all threads, both worker and master.
 */
class FastSimulationIntegration final : public IntegrationBase
{
  public:
    // Access the public-facing integration singleton
    static FastSimulationIntegration& Instance();

    // Start the run
    void BeginOfRunAction(G4Run const* run) final;

  private:
    // Tracking manager can only be created privately
    FastSimulationIntegration();
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
