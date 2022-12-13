//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/InitializeActions.cc
//---------------------------------------------------------------------------//
#include "InitializeActions.hh"

#include <G4RunManager.hh>

#include "corecel/sys/Device.hh"

#include "EventAction.hh"
#include "RunAction.hh"
#include "SetupOptions.hh"
#include "SharedParams.hh"
#include "TrackingAction.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Construct actions on each worker thread for Celeritas offloading.
 */
template<class Manager>
void initialize_actions_impl(const std::shared_ptr<const SetupOptions>& options,
                             const std::shared_ptr<SharedParams>&,
                             Manager* manager)
{
    if (Device::num_devices() > 0)
    {
        // Initialize CUDA (you'll need to use CUDA environment variables to
        // control the preferred device)
        celeritas::activate_device(Device{0});
    }

    // Run action sets up Celeritas
    manager->SetUserAction(new RunAction{options});
    // Event action saves event ID for offloading and runs queued particles at
    // end of event
    manager->SetUserAction(new EventAction{});
    // Tracking action offloads tracks to device and kills them
    manager->SetUserAction(new TrackingAction{});
}
//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Initialize actions as part of a G4VUserActionInitialization.
 *
 * The G4VUserActionInitialization::SetUserAction dispatches to
 * G4RunManager::GetRunManager(), so you should probably use that as the
 * manager argument.
 */
void InitializeActions(const std::shared_ptr<const SetupOptions>& options,
                       const std::shared_ptr<SharedParams>&       params,
                       G4RunManager*                              manager)
{
    return initialize_actions_impl(options, params, manager);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
