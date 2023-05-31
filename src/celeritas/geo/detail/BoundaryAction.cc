//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/detail/BoundaryAction.cc
//---------------------------------------------------------------------------//
#include "BoundaryAction.hh"

#include <string>

#include "corecel/Types.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "BoundaryExecutor.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace detail
{

BoundaryAction::BoundaryAction(ActionId aid,
                               std::string label,
                               std::string description)
    : ConcreteAction(aid, label, description)
{
}
//---------------------------------------------------------------------------//
/*!
 * Launch the boundary action on host.
 */
void BoundaryAction::execute(CoreParams const& params,
                             CoreStateHost& state) const
{
    TrackExecutor execute{
        params.ptr<MemSpace::native>(), state.ptr(), BoundaryExecutor{}};
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
