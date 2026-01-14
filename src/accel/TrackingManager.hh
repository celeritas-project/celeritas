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
 * Tracking managers are created by \c G4VUserPhysicsList::Construct during
 * \c G4RunManager::Initialize on each thread. The tracking manager pointer is
 * a \em thread-local part of the split-class data for a \em global G4Particle.
 * This thread-local manager points to a corresponding thread-local
 * transporter.
 *
 * Because physics initialization also happens on the master MT thread, where
 * no events are processed, a custom tracking manager \em also exists for that
 * thread. In that case, the local transporter should be null.
 *
 * \note As of Geant4 11.3, instances of this class (one per thread) will never
 * be deleted.
 *
 * \warning The physics does \em not reconstruct tracking managers on
 * subsequent runs. Therefore the \c SharedParams and \c LocalTransporter \em
 * must have lifetimes that span multiple runs (which is the case for using
 * global/thread-local).
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

    //// ACCESSORS ////

    //! Get the shared params associated with this TM
    SharedParams const* shared_params() const { return params_; }

    //! Get the thread-local transporter
    LocalTransporter* local_transporter() const { return transport_; }

  private:
    bool validated_{false};
    SharedParams const* params_{nullptr};
    LocalTransporter* transport_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
