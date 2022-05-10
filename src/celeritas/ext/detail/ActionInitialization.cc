//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include "PrimaryGeneratorAction.hh"

namespace celeritas
{
namespace detail
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
} // namespace detail
} // namespace celeritas
