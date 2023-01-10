//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/generated/BoundaryAction.cc
//! \note Auto-generated by gen-action.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "BoundaryAction.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/TrackLauncher.hh"
#include "../detail/BoundaryActionImpl.hh" // IWYU pragma: associated

namespace celeritas
{
namespace generated
{
void BoundaryAction::execute(CoreHostRef const& data) const
{
    CELER_EXPECT(data);

    MultiExceptionHandler capture_exception;
    auto launch = make_track_launcher(data, detail::boundary_track);
    #pragma omp parallel for
    for (size_type i = 0; i < data.states.size(); ++i)
    {
        CELER_TRY_HANDLE(launch(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
}

} // namespace generated
} // namespace celeritas
