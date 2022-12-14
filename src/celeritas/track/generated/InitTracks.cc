//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/generated/InitTracks.cc
//! \note Auto-generated by gen-trackinit.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "celeritas/track/detail/InitTracksLauncher.hh" // IWYU pragma: associated
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "corecel/Types.hh"

namespace celeritas
{
namespace generated
{
void init_tracks(
    const CoreHostRef& core_data,
    const size_type num_vacancies)
{
    MultiExceptionHandler capture_exception;
    detail::InitTracksLauncher<MemSpace::host> launch(core_data, num_vacancies);
    #pragma omp parallel for
    for (ThreadId::size_type i = 0; i < num_vacancies; ++i)
    {
        CELER_TRY_ELSE(launch(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
}

} // namespace generated
} // namespace celeritas
