//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/accel/offload-template/src/ActionInitialization.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VUserActionInitialization.hh>

//---------------------------------------------------------------------------//
/*!
 * Initialize all user action classes.
 */
class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    // Construct empty
    ActionInitialization();

    // Set up user actions and construct Celeritas offload interface
    void Build() const final;
};
