//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/System.cc
//---------------------------------------------------------------------------//
#include "System.hh"

#include <map>
#include <optional>

#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "celeritas/inp/System.hh"

namespace celeritas
{
namespace setup
{
//---------------------------------------------------------------------------//
/*!
 * Set up low level system properties.
 *
 * For Celeritas runs, this should be set up before anything else.
 */
void system(inp::System const& sys)
{
    // Set up environment
    auto& env = celeritas::environment();
    for (auto const& kv : sys.environment)
    {
        env.insert(kv);
        // TODO: log extant variables
    }

    // TODO: set up MPI communicator

    if (sys.device)
    {
        // TODO: if using MPI, use communicator
        activate_device();
        // NOTE: if CELER_DISABLE_DEVICE is set, device may be false
        CELER_VALIDATE(
            device(),
            << R"(failed to activate device when `sys.device` was set)");

        if (auto size = sys.device->stack_size)
        {
            set_cuda_stack_size(size);
        }
        if (auto size = sys.device->heap_size)
        {
            set_cuda_heap_size(size);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
