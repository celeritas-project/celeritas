//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/detail/BoundaryAction.cu
//---------------------------------------------------------------------------//
#include "BoundaryAction.hh"

#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "BoundaryExecutor.hh"

namespace celeritas
{
namespace detail
{

void BoundaryAction::execute(CoreParams const& params,
                             CoreStateDevice& state) const
{
    TrackExecutor execute{
        params.ptr<MemSpace::native>(), state.ptr(), BoundaryExecutor{}};

    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

}  // namespace detail
}  // namespace celeritas
