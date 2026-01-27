//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/DetectorAction.cc
//---------------------------------------------------------------------------//
#include "DetectorAction.hh"

#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "DetectorExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID.
 */
DetectorAction::DetectorAction(ActionId aid)
    : StaticConcreteAction(aid, "scoring-detector", "Score detector hits")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the detector action on host.
 */
void DetectorAction::step(CoreParams const& params, CoreStateHost& state) const
{
    TrackSlotExecutor execute{
        params.ptr<MemSpace::native>(), state.ptr(), DetectorExecutor{}};
    launch_action(state, execute);

    this->process_hits(params, state);
}

#if !CELER_USE_DEVICE
void DetectorAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
