//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/ActionDiagnosticImpl.cc
//---------------------------------------------------------------------------//
#include "ActionDiagnosticImpl.hh"

#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"

#include "ActionDiagnosticLauncher.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Tally post-step actions by particle type on host.
 */
void tally_action(HostCRef<CoreParamsData> const& params,
                  HostRef<CoreStateData> const& state,
                  HostRef<ActionDiagnosticStateData>& data)
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);
    CELER_EXPECT(data);
    MultiExceptionHandler capture_exception;
    ActionDiagnosticLauncher launch{params, state, data};
#pragma omp parallel for
    for (ThreadId::size_type i = 0; i < state.size(); ++i)
    {
        CELER_TRY_HANDLE(launch(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
