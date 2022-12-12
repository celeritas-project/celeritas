//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include "corecel/io/Logger.hh"

#include "PrimaryGeneratorAction.hh"

namespace demo_geant
{
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
    this->SetUserAction(new PrimaryGeneratorAction());
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
