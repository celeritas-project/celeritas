//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorAction.cu
//---------------------------------------------------------------------------//
#include "GeneratorAction.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/MaterialParams.hh"

#include "GeneratorExecutor.hh"
#include "OpticalGenAlgorithms.hh"
#include "../CherenkovGenerator.hh"
#include "../CherenkovParams.hh"
#include "../ScintillationGenerator.hh"
#include "../ScintillationParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to generate optical photon initializers.
 */
template<GeneratorType G>
void GeneratorAction<G>::generate(CoreParams const& core_params,
                                  CoreStateDevice& core_state) const
{
    auto& aux_state
        = get<GeneratorState<MemSpace::native>>(core_state.aux(), aux_id_);
    auto& optical_state = get<optical::CoreState<MemSpace::native>>(
        core_state.aux(), data_.optical_id);

    TrackExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        detail::GeneratorExecutor<G>{core_state.ptr(),
                                     data_.material->device_ref(),
                                     data_.shared->device_ref(),
                                     aux_state.store.ref(),
                                     optical_state.ptr(),
                                     aux_state.buffer_size,
                                     optical_state.counters()}};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(core_state, execute);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class GeneratorAction<GeneratorType::cherenkov>;
template class GeneratorAction<GeneratorType::scintillation>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
