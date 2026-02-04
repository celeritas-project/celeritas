//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/PrimaryGeneratorAction.cu
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorAction.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "detail/PrimaryGeneratorExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to generate optical photons.
 */
void PrimaryGeneratorAction::generate(CoreParams const& params,
                                      CoreStateDevice& state) const
{
    CELER_EXPECT(state.aux());

    auto const& aux_state = this->counters(*state.aux());
    size_type num_gen = min(state.sync_get_counters().num_vacancies,
                            aux_state.counters.num_pending);

    // Generate optical photons in vacant track slots
    detail::PrimaryGeneratorExecutor execute{params.ptr<MemSpace::native>(),
                                             state.ptr(),
                                             data_,
                                             params_.device_ref()};
    static ActionLauncher<decltype(execute)> const launch(*this);
    launch(num_gen, state.stream_id(), execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
