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
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/global/TrackLauncher.hh"
#include "../detail/DiscreteSelectActionImpl.hh" // IWYU pragma: associated

namespace celeritas
{
namespace generated
{
void DiscreteSelectAction::execute(CoreHostRef const& data) const
{
    CELER_EXPECT(data);

    MultiExceptionHandler capture_exception;
    auto launch = make_track_launcher(
        data.params,
        const_cast<HostRef<CoreStateData>&>(data.states),
        detail::discrete_select_track);
    #pragma omp parallel for
    for (size_type i = 0; i < data.states.size(); ++i)
    {
        CELER_TRY_HANDLE_CONTEXT(
            launch(ThreadId{i}),
            capture_exception,
            KernelContextException(data.params, data.states, ThreadId{i}, this->label()));
    }
    log_and_rethrow(std::move(capture_exception));
}

}  // namespace generated
}  // namespace celeritas
