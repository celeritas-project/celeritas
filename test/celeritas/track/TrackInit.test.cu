//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInit.test.cu
//---------------------------------------------------------------------------//
#include "TrackInit.test.hh"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "corecel/device_runtime_api.h"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void
interact_kernel(CoreStateDeviceRef const states, ITTestInputData const input)
{
    auto thread_id = KernelParamCalculator::thread_id();
    if (thread_id < states.size())
    {
        SimTrackView sim(states.sim, thread_id);

        // There may be more track slots than active tracks; only active tracks
        // should interact
        if (sim.status() != TrackStatus::inactive)
        {
            // Allow the particle to interact and create secondaries
            StackAllocator<Secondary> allocate_secondaries(
                states.physics.secondaries);

            Interactor interact(allocate_secondaries,
                                input.alloc_size[thread_id.get()],
                                input.alive[thread_id.get()]);
            auto       result = interact();

            // Save secondaries
            states.physics.state[thread_id].secondaries = result.secondaries;

            // Kill the selected tracks
            if (result.action == Interaction::Action::absorbed)
            {
                sim.status(TrackStatus::killed);
            }
        }
    }
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

void interact(CoreStateDeviceRef states, ITTestInputData input)
{
    CELER_EXPECT(states.size() > 0);
    CELER_EXPECT(states.size() == input.alloc_size.size());

    CELER_LAUNCH_KERNEL(
        interact, device().default_block_size(), states.size(), states, input);
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
