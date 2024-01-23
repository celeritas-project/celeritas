//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/HepMC3PrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "HepMC3PrimaryGeneratorAction.hh"

#include <G4Event.hh>

#include "corecel/Macros.hh"
#include "accel/ExceptionConverter.hh"
#include "accel/HepMC3PrimaryGenerator.hh"

#include "GlobalSetup.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a shared generator.
 */
HepMC3PrimaryGeneratorAction::HepMC3PrimaryGeneratorAction(SPGenerator gen)
    : generator_{std::move(gen)}
{
    CELER_EXPECT(generator_);
}

//---------------------------------------------------------------------------//
/*!
 * Generate primaries from HepMC3 input file.
 */
void HepMC3PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    ExceptionConverter call_g4exception{"celer0000"};
    CELER_TRY_HANDLE(generator_->GeneratePrimaryVertex(event),
                     call_g4exception);
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
