//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/GeneratorAction.cu
//---------------------------------------------------------------------------//
#include "GeneratorAction.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "CherenkovGenerator.hh"
#include "CherenkovParams.hh"
#include "ScintillationGenerator.hh"
#include "ScintillationParams.hh"

#include "detail/GeneratorExecutor.hh"
#include "detail/UpdateSumExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to generate optical photons.
 */
void GeneratorAction::generate(CoreParams const& params,
                               CoreStateDevice& state) const
{
    CELER_EXPECT(params.cherenkov() || params.scintillation());
    CELER_EXPECT(state.aux());

    auto& aux_state
        = get<GeneratorState<MemSpace::native>>(*state.aux(), this->aux_id());
    size_type num_gen
        = min(state.counters().num_vacancies, aux_state.counters.num_pending);
    {
        // Generate optical photons in vacant track slots
        detail::GeneratorExecutor execute{params.ptr<MemSpace::native>(),
                                          state.ptr(),
                                          params.device_ref().cherenkov,
                                          params.device_ref().scintillation,
                                          aux_state.store.ref(),
                                          aux_state.counters.buffer_size,
                                          state.counters()};
        static ActionLauncher<decltype(execute)> const launch(*this);
        launch(num_gen, state.stream_id(), execute);
    }
    {
        // Update the cumulative sum of the number of photons per distribution
        // according to how many were generated
        detail::UpdateSumExecutor execute{aux_state.store.ref(), num_gen};
        static KernelLauncher<decltype(execute)> const launch_kernel(
            "update-sum");
        launch_kernel(
            aux_state.counters.buffer_size, state.stream_id(), execute);
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
