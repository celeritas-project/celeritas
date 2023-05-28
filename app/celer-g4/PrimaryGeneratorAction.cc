//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/PrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorAction.hh"

#include <G4Event.hh>

#include "corecel/Macros.hh"
#include "accel/ExceptionConverter.hh"
#include "accel/HepMC3PrimaryGenerator.hh"

#include "GlobalSetup.hh"

using celeritas::HepMC3PrimaryGenerator;

namespace demo_geant
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Global HepMC3 file reader shared across threads.
 *
 * The first time this is called, the reader will be initialized from the
 * GlobalSetup event file argument.
 */
HepMC3PrimaryGenerator& shared_reader()
{
    static celeritas::HepMC3PrimaryGenerator reader{
        GlobalSetup::Instance()->GetEventFile()};
    return reader;
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Get the total number of events available in the HepMC3 file.
 *
 * This will load the HepMC3 file if not already active.
 */
int PrimaryGeneratorAction::NumEvents()
{
    return shared_reader().NumEvents();
}

//---------------------------------------------------------------------------//
/*!
 * Generate primaries from HepMC3 input file.
 */
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    celeritas::ExceptionConverter call_g4exception{"celer0000"};
    CELER_TRY_HANDLE(shared_reader().GeneratePrimaryVertex(event),
                     call_g4exception);
}

//---------------------------------------------------------------------------//
}  // namespace demo_geant
