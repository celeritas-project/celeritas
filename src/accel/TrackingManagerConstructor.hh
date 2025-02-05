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

    //! Null-op: particles are constructed elsewhere
    void ConstructParticle() override {}

    // Build and attach tracking manager
    void ConstructProcess() override;

  private:
    SharedParams const* shared_;
    LocalTransporterFromThread get_local_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
