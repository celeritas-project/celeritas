//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include "PrimaryGeneratorAction.hh"

namespace geant_exporter
{
//---------------------------------------------------------------------------//
/*!
 * Construct and invoke all other Geant4 classes.
 */
void ActionInitialization::Build() const
{
    this->SetUserAction(new PrimaryGeneratorAction());
}

//---------------------------------------------------------------------------//
} // namespace geant_exporter
