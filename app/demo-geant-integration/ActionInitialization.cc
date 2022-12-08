//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include "corecel/io/Logger.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct empty loads a default particle gun:
 * Default: electron with 500 MeV at the origin with direction (0, 0, 1).
 */
ActionInitialization::ActionInitialization()
    : G4VUserActionInitialization()
    , particle_gun_(
          {11, 500 * MeV, G4ThreeVector{0, 0, 1}, G4ThreeVector{0, 0, 0}})
{
    CELER_LOG_LOCAL(debug) << "ActionInitialization::"
                              "ActionInitialization with default particle "
                              "gun";
}

//---------------------------------------------------------------------------//
/*!
 * Construct with user-defined particle gun information.
 */
ActionInitialization::ActionInitialization(PGAParticleGun particle_gun)
    : G4VUserActionInitialization(), particle_gun_(particle_gun)
{
    CELER_LOG_LOCAL(debug) << "ActionInitialization::"
                              "ActionInitialization with user-defined "
                              "particle gun";
}

//---------------------------------------------------------------------------//
/*!
 * Construct actions on manager thread.
 *
 * This is *only* called if using multithreaded Geant4.
 */
void ActionInitialization::BuildForMaster() const
{
    CELER_LOG_LOCAL(debug) << "ActionInitialization::BuildForMaster";
}

//---------------------------------------------------------------------------//
/*!
 * Construct actions on each worker thread.
 */
void ActionInitialization::Build() const
{
    CELER_LOG_LOCAL(debug) << "ActionInitialization::Build";

    // Initialize primary generator
    this->SetUserAction(new PrimaryGeneratorAction(particle_gun_));
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
