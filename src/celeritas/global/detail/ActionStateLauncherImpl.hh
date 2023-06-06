//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/ActionStateLauncherImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//

Range<ThreadId> compute_launch_params(ActionId action,
                                      CoreParams const& params,
                                      CoreState<MemSpace::device> const& state,
                                      TrackOrder expected);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas