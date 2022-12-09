//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"

#include "EventAction.hh"
#include "RunAction.hh"
#include "TrackingAction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Celeritas setup options.
 */
ActionInitialization::ActionInitialization(SPCOptions   options,
                                           UPUserAction action)
    : options_(std::move(options)), action_(std::move(action))
{
    CELER_EXPECT(options_);
    params_ = std::make_shared<SharedParams>();
}

//---------------------------------------------------------------------------//
/*!
 * Construct actions on manager thread.
 *
 * This is *only* called if using multithreaded Geant4.
 */
void ActionInitialization::BuildForMaster() const
{
    CELER_LOG_LOCAL(debug) << "ActionInitialization::BuildForMaster";

    this->SetUserAction(new RunAction(options_, params_));

    if (action_)
    {
        action_->BuildForMaster();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct actions on each worker thread.
 */
void ActionInitialization::Build() const
{
    CELER_LOG_LOCAL(debug) << "ActionInitialization::Build";

    if (Device::num_devices() > 0)
    {
        // Initialize CUDA (you'll need to use CUDA environment variables to
        // control the preferred device)
        celeritas::activate_device(Device{0});
        // TODO: these should be user configurable because they're geometry
        // dependent
        celeritas::set_cuda_stack_size(32768);
        celeritas::set_cuda_heap_size(12582912);
    }

    // Run action sets up Celeritas
    this->SetUserAction(new RunAction{options_, params_});
    // Event action saves event ID for offloading and runs queued particles at
    // end of event
    this->SetUserAction(new EventAction{});
    // Tracking action offloads tracks to device and kills them
    this->SetUserAction(new TrackingAction{});

    if (action_)
    {
        action_->Build();
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
