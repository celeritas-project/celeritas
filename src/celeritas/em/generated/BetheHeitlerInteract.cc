//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/BetheHeitlerInteract.cc
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "BetheHeitlerInteract.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/em/launcher/BetheHeitlerLauncher.hh" // IWYU pragma: associated
#include "celeritas/phys/InteractionLauncher.hh"

using celeritas::MemSpace;

namespace celeritas
{
namespace generated
{
void bethe_heitler_interact(
    celeritas::BetheHeitlerHostRef const& model_data,
    celeritas::CoreRef<MemSpace::host> const& core_data)
{
    CELER_EXPECT(core_data);
    CELER_EXPECT(model_data);

    celeritas::MultiExceptionHandler capture_exception;
    auto launch = celeritas::make_interaction_launcher(
        core_data,
        model_data,
        celeritas::bethe_heitler_interact_track);
    #pragma omp parallel for
    for (celeritas::size_type i = 0; i < core_data.states.size(); ++i)
    {
        CELER_TRY_HANDLE(launch(celeritas::ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
}

}  // namespace generated
}  // namespace celeritas
