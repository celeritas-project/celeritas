//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/generated/DiscreteSelectAction.cc
//! \note Auto-generated by gen-action.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "DiscreteSelectAction.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/global/TrackLauncher.hh"
#include "../detail/DiscreteSelectActionImpl.hh" // IWYU pragma: associated

namespace celeritas
{
namespace generated
{
void DiscreteSelectAction::execute(CoreParams const& params, StateHostRef& state) const
{
    CELER_EXPECT(state);

    MultiExceptionHandler capture_exception;
    TrackLauncher launch{params.ref<MemSpace::native>(), state, detail::discrete_select_track};
    #pragma omp parallel for
    for (size_type i = 0; i < state.size(); ++i)
    {
        CELER_TRY_HANDLE_CONTEXT(
            launch(ThreadId{i}),
            capture_exception,
            KernelContextException(params.ref<MemSpace::host>(), state, ThreadId{i}, this->label()));
    }
    log_and_rethrow(std::move(capture_exception));
}

}  // namespace generated
}  // namespace celeritas
