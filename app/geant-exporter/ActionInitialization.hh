//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ActionInitialization.hh
//! \brief Invoke UserAction type classes
//---------------------------------------------------------------------------//
#pragma once

#include <G4VUserActionInitialization.hh>

namespace geant_exporter
{
//---------------------------------------------------------------------------//
/*!
 * Initialize Geant4.
 */
class ActionInitialization : public G4VUserActionInitialization
{
  public:
    void Build() const override;
};

//---------------------------------------------------------------------------//
} // namespace geant_exporter
