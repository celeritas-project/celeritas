//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/CombinedBremInteract.cc
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "CombinedBremInteract.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/em/executor/CombinedBremExecutor.hh" // IWYU pragma: associated
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/phys/InteractionExecutor.hh"

using celeritas::MemSpace;

namespace celeritas
{
namespace generated
{
void combined_brem_interact(
    celeritas::CoreParams const& params,
    celeritas::CoreState<MemSpace::host>& state,
    celeritas::CombinedBremHostRef const& model_data)
{
    CELER_EXPECT(model_data);

    celeritas::MultiExceptionHandler capture_exception;
    auto execute = celeritas::make_interaction_executor(
        params.ptr<MemSpace::native>(), state.ptr(),
        celeritas::combined_brem_interact_track, model_data);
    #pragma omp parallel for
    for (celeritas::size_type i = 0; i < state.size(); ++i)
    {
        CELER_TRY_HANDLE_CONTEXT(
            execute(ThreadId{i}),
            capture_exception,
            KernelContextException(params.ref<MemSpace::host>(), state.ref(), ThreadId{i}, "combined_brem"));
    }
    log_and_rethrow(std::move(capture_exception));
}

}  // namespace generated
}  // namespace celeritas
