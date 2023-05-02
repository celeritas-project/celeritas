//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/LivermorePEInteract.cc
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "LivermorePEInteract.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/em/launcher/LivermorePELauncher.hh" // IWYU pragma: associated
#include "celeritas/phys/InteractionLauncher.hh"

using celeritas::MemSpace;

namespace celeritas
{
namespace generated
{
void livermore_pe_interact(
    celeritas::LivermorePEHostRef const& model_data,
    celeritas::CoreParams const& params,
    celeritas::CoreState<MemSpace::host>& state)
{
    CELER_EXPECT(model_data);

    celeritas::MultiExceptionHandler capture_exception;
    auto launch = celeritas::make_interaction_launcher(
        params.ref<MemSpace::native>(), state.ref(), model_data,
        celeritas::livermore_pe_interact_track);
    #pragma omp parallel for
    for (celeritas::size_type i = 0; i < state.size(); ++i)
    {
        CELER_TRY_HANDLE_CONTEXT(
            launch(ThreadId{i}),
            capture_exception,
            KernelContextException(params.ref<MemSpace::host>(), state.ref(), ThreadId{i}, "livermore_pe"));
    }
    log_and_rethrow(std::move(capture_exception));
}

}  // namespace generated
}  // namespace celeritas
