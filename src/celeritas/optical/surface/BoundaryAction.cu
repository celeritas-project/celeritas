//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/BoundaryAction.cu
//---------------------------------------------------------------------------//
#include "BoundaryAction.hh"

#include "geocel/SurfaceParams.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "detail/InitBoundaryExecutor.hh"
#include "detail/PostBoundaryExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch kernel with device data.
 */
template<class E>
void BoundaryAction<E>::step(CoreParams const& params,
                             CoreStateDevice& state) const
{
    auto execute = make_action_thread_executor(
        params.ptr<MemSpace::native>(), state.ptr(), this->action_id(), E{});

    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class BoundaryAction<detail::InitBoundaryExecutor>;
template class BoundaryAction<detail::PostBoundaryExecutor>;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
