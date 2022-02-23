//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInit.test.cu
//---------------------------------------------------------------------------//
#include "TrackInit.test.hh"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include "base/device_runtime_api.h"
#include "comm/Device.hh"
#include "base/KernelParamCalculator.device.hh"

namespace celeritas_test
{
using namespace celeritas;

//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void interact_kernel(StateDeviceRef states, ITTestInputData input)
{
    auto thread_id = celeritas::KernelParamCalculator::thread_id();
    if (thread_id < states.size())
    {
        SimTrackView sim(states.sim, thread_id);

        // There may be more track slots than active tracks; only active tracks
        // should interact
        if (sim.alive())
        {
            // Allow the particle to interact and create secondaries
            StackAllocator<Secondary> allocate_secondaries(states.secondaries);
            Interactor                interact(allocate_secondaries,
                                input.alloc_size[thread_id.get()],
                                input.alive[thread_id.get()]);
            states.interactions[thread_id] = interact();
            CELER_ASSERT(states.interactions[thread_id]);

            // Kill the selected tracks
            if (!input.alive[thread_id.get()])
            {
                sim.alive(false);
            }
        }
    }
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

void interact(StateDeviceRef states, ITTestInputData input)
{
    CELER_EXPECT(states.size() > 0);
    CELER_EXPECT(states.size() == input.alloc_size.size());

    CELER_LAUNCH_KERNEL(interact,
                        celeritas::device().default_block_size(),
                        states.size(),
                        states,
                        input);
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
