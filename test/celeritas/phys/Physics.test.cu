//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Physics.test.cu
//---------------------------------------------------------------------------//
#include "Physics.test.hh"

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void phys_test_kernel(PTestInput const inp)
{
    auto tid = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};
    if (tid.get() >= inp.states.size())
        return;

    auto const& init = inp.inits[tid];
    ParticleTrackView par(inp.par_params, inp.par_states, tid);
    par = ParticleTrackView::Initializer_t{init.particle, init.energy};
    PhysicsTrackView phys(inp.params, inp.states, par, init.mat, tid);
    PhysicsStepView step(inp.params, inp.states, tid);

    phys = PhysicsTrackInitializer{};
    inp.result[tid.get()]
        = native_value_to<units::CmLength>(calc_step(phys, step, init.energy))
              .value();
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void phys_cuda_test(PTestInput const& input)
{
    CELER_ASSERT(input.inits.size() == input.states.size());

    CELER_LAUNCH_KERNEL(phys_test, input.states.size(), 0, input);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
