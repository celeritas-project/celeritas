//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/PrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorAction.hh"

#include "corecel/Macros.hh"
#include "accel/ExceptionConverter.hh"

#include "GlobalSetup.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Generate primaries from HepMC3 input file.
 */
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    celeritas::ExceptionConverter call_g4exception{"celer0000"};
    CELER_TRY_ELSE(this->Reader().GeneratePrimaryVertex(event),
                   call_g4exception);
}

//---------------------------------------------------------------------------//
/*!
 * Global HepMC3 file reader shared across threads.
 *
 * The first time this is called, the reader will be initialized from the
 * GlobalSetup event file argument.
 */
celeritas::HepMC3Reader& PrimaryGeneratorAction::Reader()
{
    static celeritas::HepMC3Reader reader{
        GlobalSetup::Instance()->GetEventFile()};
    return reader;
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
