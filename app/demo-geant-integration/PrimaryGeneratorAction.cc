//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/PrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorAction.hh"

#include <G4ParticleTable.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>

#include "HepMC3Reader.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Generate primaries from HepMC3 input file.
 */
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    HepMC3Reader::instance()->GeneratePrimaryVertex(event);
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
