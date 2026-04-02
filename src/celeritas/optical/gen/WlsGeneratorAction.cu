//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/WlsGeneratorAction.cu
//---------------------------------------------------------------------------//
#include "WlsGeneratorAction.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"
#include "celeritas/optical/model/WavelengthShiftModel.hh"

#include "WavelengthShiftGenerator.hh"

#include "detail/WlsGeneratorExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to generate optical WLS photons.
 */
void WlsGeneratorAction::generate(CoreParams const& params,
                                  CoreStateDevice& state) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state = get<WlsGeneratorState<MemSpace::native>>(*state.aux(),
                                                               this->aux_id());
    size_type num_gen = min(state.sync_get_counters().num_vacancies,
                            aux_state.counters.num_pending);

    // Generate optical photons in vacant track slots
    detail::WlsGeneratorExecutor execute{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        wls_ ? wls_->device_ref() : NativeCRef<WavelengthShiftData>{},
        wls2_ ? wls2_->device_ref() : NativeCRef<WavelengthShiftData>{},
        aux_state.store.ref(),
        aux_state.counters.buffer_size};
    static ActionLauncher<decltype(execute)> const launch(*this);
    launch(num_gen, state.stream_id(), execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
