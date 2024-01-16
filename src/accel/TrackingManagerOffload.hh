//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerOffload.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VTrackingManager.hh>

namespace celeritas
{
class SharedParams;
class LocalTransporter;

//---------------------------------------------------------------------------//
/*!
 * Offload tracks to Celeritas via the per-particle G4VTrackingManager
 * interface
 */
class TrackingManagerOffload final : public G4VTrackingManager
{
  public:
    // Construct with shared (across threads) params, and thread-local
    // transporter.
    TrackingManagerOffload(SharedParams const* params, LocalTransporter* local);

    // Prepare cross-section tables for rebuild (e.g. if new materials have
    // been defined).
    void PreparePhysicsTable(G4ParticleDefinition const&) override;

    // Rebuild physics cross-section tables (e.g. if new materials have been
    // defined).
    void BuildPhysicsTable(G4ParticleDefinition const&) override;

    // Hand over passed track to this tracking manager.
    void HandOverOneTrack(G4Track* aTrack) override;

    // Complete processing of any buffered tracks.
    void FlushEvent() override;

  private:
    SharedParams const* params_{nullptr};
    LocalTransporter* transport_{nullptr};
};
}  // namespace celeritas
