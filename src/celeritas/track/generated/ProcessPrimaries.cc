//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/generated/ProcessPrimaries.cc
//! \note Auto-generated by gen-trackinit.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include <utility>

#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "corecel/Types.hh"
#include "celeritas/track/detail/ProcessPrimariesLauncher.hh" // IWYU pragma: associated

namespace celeritas
{
namespace generated
{
void process_primaries(
    CoreHostRef const& core_data,
    Span<const Primary> const primaries)
{
    MultiExceptionHandler capture_exception;
    detail::ProcessPrimariesLauncher<MemSpace::host> launch(core_data, primaries);
    #pragma omp parallel for
    for (ThreadId::size_type i = 0; i < primaries.size(); ++i)
    {
        CELER_TRY_HANDLE(launch(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
}

}  // namespace generated
}  // namespace celeritas
