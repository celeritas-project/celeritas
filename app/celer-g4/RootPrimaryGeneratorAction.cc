//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RootPrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "RootPrimaryGeneratorAction.hh"

#include "corecel/Macros.hh"
#include "accel/ExceptionConverter.hh"
#include "accel/RootPrimaryGenerator.hh"

#include "GlobalSetup.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a shared generator.
 */
RootPrimaryGeneratorAction::RootPrimaryGeneratorAction(SPGenerator gen)
    : generator_(std::move(gen))
{
    CELER_EXPECT(generator_);
}

//---------------------------------------------------------------------------//
/*!
 * Generate primaries from ROOT input file with offloaded primary data.
 */
void RootPrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    ExceptionConverter call_g4exception{"celer0000"};
    CELER_TRY_HANDLE(generator_->GeneratePrimaryVertex(event),
                     call_g4exception);
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
