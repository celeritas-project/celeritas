//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/TrivialFacetNormalAction.cc
//---------------------------------------------------------------------------//
#include "TrivialFacetNormalAction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//

TrivialFacetNormalAction::TrivialFacetNormalAction(ActionId aid)
    : ConcreteAction(
          aid, "sample-normal-trivial", "Sample trivial surface normal")
{
}

void TrivialFacetNormalAction::step(CoreParams const& params,
                                    CoreStateHost& state) const
{
    auto execute = make_action_thread_executor(params.ptr<MemSpace::native>(),
                                               state.ptr(),
                                               this->action_id(),
                                               TrivialFacetNormalExecutor{});
    launch_action(execute);
}

#if !CELER_USE_DEVICE
void TrivialFacetNormalAction::step(CoreParams const& params,
                                    CoreStateDevice& state) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
