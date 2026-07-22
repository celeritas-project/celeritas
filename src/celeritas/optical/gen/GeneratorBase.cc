//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/GeneratorBase.cc
//---------------------------------------------------------------------------//
#include "GeneratorBase.hh"

#include "corecel/Assert.hh"
#include "corecel/data/AuxStateVec.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/TrackExecutor.hh"
#include "celeritas/optical/action/ActionLauncher.hh"

#include "detail/UpdatePendingExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with IDs, label, and description.
 */
GeneratorBase::GeneratorBase(
    ActionId id,
    AuxId aux_id,
    GeneratorId gen_id,
    std::string_view label,
    std::string_view description) noexcept(!CELERITAS_DEBUG)
    : sad_{id, label, description}, aux_id_(aux_id), gen_id_(gen_id)
{
    CELER_EXPECT(aux_id_);
    CELER_EXPECT(gen_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Get generator counters (mutable).
 */
GeneratorStateBase& GeneratorBase::counters(AuxStateVec& aux) const
{
    return dynamic_cast<GeneratorStateBase&>(aux.at(aux_id_));
}

//---------------------------------------------------------------------------//
/*!
 * Get generator counters.
 */
GeneratorStateBase const& GeneratorBase::counters(AuxStateVec const& aux) const
{
    return dynamic_cast<GeneratorStateBase const&>(aux.at(aux_id_));
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to update the number of pending optical photons.
 */
void GeneratorBase::update_pending(
    CoreParams const& params, CoreStateHost& state, size_type num_pending) const
{
    // Update the number of pending optical photons
    auto execute_thread = make_single_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::UpdatePendingExecutor{num_pending});
    launch_action(1, execute_thread);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void GeneratorBase::update_pending(
    CoreParams const&, CoreStateDevice&, size_type) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
