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

#include "detail/IntegrationSingleton.hh"

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
 * before the simulation begins.  By default this uses the \c
 * celeritas::TrackingManagerIntegration helper:
 * \code
    auto* physics_list = new FTFP_BERT;
    physics_list->RegisterPhysics(new TrackingManagerConstructor{
        &TrackingManagerIntegration::Instance()});
   \endcode
 *
 * but for manual integration it can be constructed with a function to get a
 * reference to the thread-local \c LocalTransporter from the Geant4 thread ID:
 * \code
    auto* physics_list = new FTFP_BERT;
    physics_list->RegisterPhysics(new TrackingManagerConstructor{
        shared_params, [](int){ return &local_transporter; });
   \endcode
 *
 * \note If Celeritas is globally disabled, it will not add the track manager.
 * If Celeritas is configured to "kill offload" mode (for testing maximum
 * theoretical performance) then the tracking manager will be added but will
 * not send the tracks to Celeritas: it will simply kill them.
 */
class TrackingManagerConstructor final : public G4VPhysicsConstructor
{
  public:
    //!@{
    //! \name Type aliases
    using LocalTransporterFromThread = std::function<LocalTransporter*(int)>;
    using VecG4PD = SetupOptions::VecG4PD;
    //!@}

  public:
    // Construct name and mode
    TrackingManagerConstructor(SharedParams const* shared,
                               LocalTransporterFromThread get_local);

    // Construct from tracking manager integration
    explicit TrackingManagerConstructor(TrackingManagerIntegration* tmi);

    //! Build list of particles to be offloaded
    void ConstructParticle() override;

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
    VecG4PD offload_particles_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
