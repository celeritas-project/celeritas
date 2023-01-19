//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/HitClasses.hh
//---------------------------------------------------------------------------//
#include <G4VHit.hh>

#include "HitRootIO.hh"
#include "SensitiveHit.hh"
//---------------------------------------------------------------------------//
/*!
 * Declaration of classes for generating ROOT dictionaries
 */
demo_geant::HitRootEvent e;
std::vector<G4VHit*> vh;
std::vector<demo_geant::SensitiveHit*> vsh;
std::map<std::string, std::vector<G4VHit*>> mhc;
#undef __G4String
