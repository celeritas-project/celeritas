//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4Version.hh>
#if G4VERSION_NUMBER < 1100
#    error "Tracking manager offload requires Geant4 11.0 or higher"
#endif

#include <G4VTrackingManager.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//

class SharedParams;
class LocalTransporter;

//---------------------------------------------------------------------------//
/*!
 * Offload to Celeritas via the per-particle Geant4 "tracking manager".
 *
 * Tracking managers are to be created during worker action initialization and
 * are thus thread-local.  Construction/addition to \c G4ParticleDefinition
 * appears to take place on the master thread, typically
 * in the ConstructProcess method, but the tracking manager pointer is part of
 * the split-class data for the particle. It's observed that different threads
 * have distinct pointers to a LocalTransporter instance, and that these match
 * those of the global thread-local instances in test problems.
 *
 * \note As of Geant4 11.3, instances of this class (one per thread) will never
 * be deleted.
 */
class TrackingManager final : public G4VTrackingManager
{
  public:
    // Construct with shared (across threads) params, and thread-local
    // transporter.
    TrackingManager(SharedParams const* params, LocalTransporter* local);

    // Prepare cross-section tables for rebuild (e.g. if new materials have
    // been defined).
    void PreparePhysicsTable(G4ParticleDefinition const&) final;

    // Rebuild physics cross-section tables (e.g. if new materials have been
    // defined).
    void BuildPhysicsTable(G4ParticleDefinition const&) final;

    // Hand over passed track to this tracking manager.
    void HandOverOneTrack(G4Track* aTrack) final;

    // Complete processing of any buffered tracks.
    void FlushEvent() final;

  private:
    SharedParams const* params_{nullptr};
    LocalTransporter* transport_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
