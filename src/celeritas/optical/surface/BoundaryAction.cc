//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/BoundaryAction.cc
//---------------------------------------------------------------------------//
#include "BoundaryAction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID.
 */
BoundaryAction::BoundaryAction(ActionId aid)
    : ConcreteAction(aid, "geo-boundary", "cross a geometry boundary")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the boundary action on host.
 */
void BoundaryAction::step(CoreParams const& params, CoreStateHost& state) const
{
    auto const& surface = params.surface_physics();

    // Have each surface model choose the appropriate models for each step
    for (auto const& model : surface.models())
    {
        model->select_action_sequence(params, state);
    }

    // Run facet normal sequence
    for (auto const& facet_normal_action : surface.facet_normal_actions())
    {
        facet_normal_action->step(params, state);
    }

    // Run calculate reflectivity sequence
    for (auto const& calc_reflectivity_action :
         surface.calculate_reflectivity_actions())
    {
        calc_reflectivity_action->step(params, state);
    }

    // Select reflection / transmission / absorption
    //

    auto execute = make_action_thread_executor(params.ptr<MemSpace::native>(),
                                               state.ptr(),
                                               this->action_id(),
                                               detail::BoundaryExecutor{});
    return launch_action(state, execute);
}

#if !CELER_USE_DEVICE
void BoundaryAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
