//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
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
interact_kernel(CoreRef<MemSpace::device> data, ITTestInputData const input)
{
    auto slot_id
        = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};
    if (slot_id < data.states.size())
    {
        SimTrackView sim(data.params.sim, data.states.sim, slot_id);

        // There may be more track slots than active tracks; only active tracks
        // should interact
        if (sim.status() != TrackStatus::inactive)
        {
            // Allow the particle to interact and create secondaries
            StackAllocator<Secondary> allocate_secondaries(
                data.states.physics.secondaries);

            Interactor interact(allocate_secondaries,
                                input.alloc_size[slot_id.get()],
                                input.alive[slot_id.get()]);
            auto result = interact();

            // Save secondaries
            data.states.physics.state[slot_id].secondaries = result.secondaries;

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

void interact(CoreRef<MemSpace::device> data, ITTestInputData input)
{
    CELER_EXPECT(data.states.size() > 0);
    CELER_EXPECT(data.states.size() == input.alloc_size.size());

    CELER_LAUNCH_KERNEL(interact,
                        device().default_block_size(),
                        data.states.size(),
                        data,
                        input);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
