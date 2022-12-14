//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include <G4RunManager.hh>

#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "accel/InitializeActions.hh"

#include "GlobalSetup.hh"
#include "PrimaryGeneratorAction.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct global data to be shared across Celeritas workers.
 */
ActionInitialization::ActionInitialization()
{
    // Create params to be shared across worker threads
    params_ = std::make_shared<celeritas::SharedParams>();
    // Make global setup commands available to UI
    GlobalSetup::Instance();
}

//---------------------------------------------------------------------------//
/*!
 * Construct actions on each worker thread.
 */
void ActionInitialization::Build() const
{
    CELER_LOG_LOCAL(status) << "Constructing user actions on worker threads";

    // Initialize primary generator
    this->SetUserAction(new PrimaryGeneratorAction());

    // Initialize celeritas and add user actions
    celeritas::InitializeActions(GlobalSetup::Instance()->GetSetupOptions(),
                                 params_,
                                 G4RunManager::GetRunManager());

    if (unsigned int sz = GlobalSetup::Instance()->GetCudaStackSize())
    {
        celeritas::set_cuda_stack_size(sz);
    }
    if (unsigned int sz = GlobalSetup::Instance()->GetCudaHeapSize())
    {
        celeritas::set_cuda_heap_size(sz);
    }
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
