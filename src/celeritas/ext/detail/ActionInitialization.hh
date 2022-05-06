//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/ActionInitialization.hh
//! Invoke UserAction type classes.
//---------------------------------------------------------------------------//
#pragma once

#include <G4VUserActionInitialization.hh>

namespace celeritas
{
namespace detail
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
} // namespace detail
} // namespace celeritas
