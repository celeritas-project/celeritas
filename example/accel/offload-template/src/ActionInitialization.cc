//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/accel/offload-template/src/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include "EventAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "TrackingAction.hh"

//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
ActionInitalization::ActionInitalization() : G4VUserActionInitialization() {}

//---------------------------------------------------------------------------//
/*!
 * Set up all user actions.
 */
void ActionInitalization::Build() const
{
    this->SetUserAction(new PrimaryGeneratorAction());
    this->SetUserAction(new RunAction());
    this->SetUserAction(new EventAction());
    this->SetUserAction(new TrackingAction());
}
