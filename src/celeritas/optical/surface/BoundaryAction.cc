//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/BoundaryAction.cc
//---------------------------------------------------------------------------//
#include "BoundaryAction.hh"

#include "geocel/SurfaceParams.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "detail/InitBoundaryExecutor.hh"
#include "detail/PostBoundaryExecutor.hh"

namespace celeritas
{
namespace optical
{
namespace
{
//---------------------------------------------------------------------------//

template<class E>
struct BoundaryActionTraits;

template<>
struct BoundaryActionTraits<detail::InitBoundaryExecutor>
{
    constexpr static char const* action_name = "optical-boundary-init";
    constexpr static char const* action_desc
        = "Initialize optical boundary crossing action";
};

template<>
struct BoundaryActionTraits<detail::PostBoundaryExecutor>
{
    constexpr static char const* action_name = "optical-boundary-post";
    constexpr static char const* action_desc
        = "Finalize optical boundary crossing action";
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct the boundary action from an action ID.
 */
template<class E>
BoundaryAction<E>::BoundaryAction(ActionId aid)
    : ConcreteAction(aid,
                     BoundaryActionTraits<E>::action_name,
                     BoundaryActionTraits<E>::action_desc)
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch kernel with host data.
 */
template<class E>
void BoundaryAction<E>::step(CoreParams const& params,
                             CoreStateHost& state) const
{
    launch_action(state,
                  make_action_thread_executor(params.ptr<MemSpace::native>(),
                                              state.ptr(),
                                              this->action_id(),
                                              E{}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute kernel with device data.
 */
#if !CELER_USE_DEVICE
template<class E>
void BoundaryAction<E>::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class BoundaryAction<detail::InitBoundaryExecutor>;
template class BoundaryAction<detail::PostBoundaryExecutor>;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
