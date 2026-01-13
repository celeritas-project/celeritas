//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceSteppingAction.cc
//---------------------------------------------------------------------------//
#include "SurfaceSteppingAction.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "SurfaceModel.hh"
#include "SurfacePhysicsParams.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct surface stepping action from ID.
 */
SurfaceSteppingAction::SurfaceSteppingAction(ActionId aid)
    : ConcreteAction(aid,
                     "optical-surface-stepping",
                     "step through optical surface actions")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch kernel with host data.
 */
void SurfaceSteppingAction::step(CoreParams const& params,
                                 CoreStateHost& state) const
{
    this->step_impl<MemSpace::host>(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Launch kernel with device data.
 */
void SurfaceSteppingAction::step(CoreParams const& params,
                                 CoreStateDevice& state) const
{
    this->step_impl<MemSpace::device>(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Launch kernel in the given \c MemSpace.
 *
 * Loops over all sub-steps in order, launching actions from each sub-step.
 * Currently the loop is limited to a single iteration, but this may be changed
 * in the future.
 */
template<MemSpace M>
void SurfaceSteppingAction::step_impl(CoreParams const& params,
                                      CoreState<M>& state) const
{
    constexpr size_type num_iterations = 1;

    for ([[maybe_unused]] auto iteration : range(num_iterations))
    {
        for (auto substep : range(SurfacePhysicsOrder::size_))
        {
            for (auto const& model : params.surface_physics()->models(substep))
            {
                model->step(params, state);
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
