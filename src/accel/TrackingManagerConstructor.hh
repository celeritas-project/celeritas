//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerConstructor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <G4ParticleDefinition.hh>
#include <G4VPhysicsConstructor.hh>

#include "corecel/cont/Span.hh"

namespace celeritas
{
class LocalTransporter;
class SharedParams;
class TrackingManagerIntegration;

//---------------------------------------------------------------------------//
/*!
 * Construct a Celeritas tracking manager that offloads EM tracks.
 *
 * This should be composed with your physics list after it is constructed,
 * before the simulation begins.
 *
 * \code
    auto* physics_list = new FTFP_BERT;
    physics_list->RegisterPhysics(new TrackingManagerConstructor{
        shared_params, [](int){ return &local_transporter; });
   \endcode
 *
 * For simpler integration, use celeritas::TrackingManagerIntegration:
 * \code
    auto* physics_list = new FTFP_BERT;
    physics_list->RegisterPhysics(new TrackingManagerConstructor{
        &TrackingManagerIntegration::Instance()});
   \endcode
 *
 *
 * The second argument is a function to get a reference to the thread-local \c
 * LocalTransporter from the Geant4 thread ID.
 *
 * If Celeritas is globally disabled, it will not add the track manager.
 */
class TrackingManagerConstructor final : public G4VPhysicsConstructor
{
  public:
    //!@{
    //! \name Type aliases
    using LocalTransporterFromThread = std::function<LocalTransporter*(int)>;
    //!@}

  public:
    // Get a list of supported particles
    static Span<G4ParticleDefinition* const> OffloadParticles();

    // Construct name and mode
    TrackingManagerConstructor(SharedParams const* shared,
                               LocalTransporterFromThread get_local);

    // Construct from tracking manager integration
    explicit TrackingManagerConstructor(TrackingManagerIntegration* tmi);

    //! Null-op: particles are constructed elsewhere
    void ConstructParticle() override {}

    // Build and attach tracking manager
    void ConstructProcess() override;

    //// ACCESSORS ////

    //! Get the shared params associated with this TM
    SharedParams const* shared_params() const { return shared_; }

    // Get the local transporter associated with the current thread ID
    LocalTransporter* get_local_transporter() const;

  private:
    SharedParams const* shared_{nullptr};
    LocalTransporterFromThread get_local_{};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
