//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadAction.cu
//---------------------------------------------------------------------------//
#include "OffloadAction.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/MaterialParams.hh"

#include "CherenkovOffloadExecutor.hh"
#include "OpticalGenAlgorithms.hh"
#include "ScintOffloadExecutor.hh"
#include "../CherenkovParams.hh"
#include "../ScintillationParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to generate optical distribution data post-step.
 */
template<GeneratorType G>
void OffloadAction<G>::offload(CoreParams const& core_params,
                               CoreStateDevice& core_state) const
{
    auto& step = core_state.aux_data<OffloadStepStateData>(data_.step_id);
    auto& gen_state = get<GeneratorState<MemSpace::native>>(core_state.aux(),
                                                            data_.gen_id);
    TrackExecutor execute{core_params.ptr<MemSpace::native>(),
                          core_state.ptr(),
                          Executor{data_.material->device_ref(),
                                   data_.shared->device_ref(),
                                   gen_state.store.ref(),
                                   step,
                                   gen_state.counters.buffer_size}};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(core_state, execute);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class OffloadAction<GeneratorType::cherenkov>;
template class OffloadAction<GeneratorType::scintillation>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
