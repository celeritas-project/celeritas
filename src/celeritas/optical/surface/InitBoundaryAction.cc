//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/InitBoundaryAction.cc
//---------------------------------------------------------------------------//
#include "InitBoundaryAction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
InitBoundaryAction::InitBoundaryAction(ActionId aid)
    : ConcreteAction(aid,
                     "optical-boundary-init",
                     "Initialize optical boundary crossing action")
{
}

void InitBoundaryAction::step(CoreParams const& params,
                              CoreStateHost& state) const
{
}

#if !CELER_USE_DEVICE
void InitBoundaryAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
