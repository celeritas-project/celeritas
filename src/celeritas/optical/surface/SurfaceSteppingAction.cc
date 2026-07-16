//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceSteppingAction.cc
//---------------------------------------------------------------------------//
#include "SurfaceSteppingAction.hh"

#include <string_view>

#include "corecel/sys/ScopedProfiling.hh"
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
    : ConcreteAction(aid, "surface-physics", "apply optical surface physics")
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
    auto const& physics = *params.surface_physics();
    constexpr size_type num_iterations = 1;

    for ([[maybe_unused]] auto iteration : range(num_iterations))
    {
        for (auto substep : range(SurfacePhysicsOrder::size_))
        {
            auto const& models = physics.models(substep);
            if (models.empty())
                continue;

            ScopedProfiling profile_this_{
                std::string_view(to_cstring(substep))};
            for (auto const& model : models)
            {
                model->step(params, state);
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
