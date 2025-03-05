//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/CaloAction.cc
//---------------------------------------------------------------------------//
#include "CaloAction.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.hh"
#include "TrackSlotExecutor.hh"

#include "detail/CaloExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID.
 */
CaloAction::CaloAction(ActionId aid)
    : StaticConcreteAction(aid, "track-calo", "contribute to detector")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the boundary action on host.
 */
void CaloAction::step(CoreParams const& params, CoreStateHost& state) const
{
    auto detector_ids = params.detector_ids();
    auto detect_and_update
        = [](CoreTrackView& track, std::vector<VolumeId>& detector_ids) {
              detail::CaloExecutor{}(track, detector_ids);
          };
    auto execute = make_active_thread_executor(
        params.ptr<MemSpace::native>(), state.ptr(), detect_and_update);
    return launch_action(state, execute);
}

#if !CELER_USE_DEVICE
void CaloAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas