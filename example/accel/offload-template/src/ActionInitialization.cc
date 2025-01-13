//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/accel/offload-template/src/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include "Celeritas.hh"
#include "EventAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "TrackingAction.hh"

//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
ActionInitialization::ActionInitialization() : G4VUserActionInitialization() {}

//---------------------------------------------------------------------------//
/*!
 * Set up all user actions and construct Celeritas' offloading interface.
 */
void ActionInitialization::Build() const
{
    CelerSimpleOffload().Build(
        &CelerSetupOptions(), &CelerSharedParams(), &CelerLocalTransporter());

    this->SetUserAction(new RunAction());
    this->SetUserAction(new EventAction());
    this->SetUserAction(new TrackingAction());
    this->SetUserAction(new PrimaryGeneratorAction());
}
