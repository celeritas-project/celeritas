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
#include "accel/HepMC3Reader.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Generate primaries from HepMC3 input file.
 */
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    celeritas::ExceptionConverter call_g4exception{"celer0000"};
    CELER_TRY_ELSE(
        celeritas::HepMC3Reader::Instance()->GeneratePrimaryVertex(event),
        call_g4exception);
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
