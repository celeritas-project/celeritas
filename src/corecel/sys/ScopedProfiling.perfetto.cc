//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.perfetto.cc
//! \brief The roctx implementation of \c ScopedProfiling
//---------------------------------------------------------------------------//
#include "ScopedProfiling.hh"

#include <perfetto.h>

#include "corecel/io/Logger.hh"

#include "Environment.hh"

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace celeritas
{
void initialize_perfetto(perfetto::TracingInitArgs const& args)
{
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
}

bool ScopedProfiling::use_profiling()
{
    static bool const result = [] {
        if (!celeritas::getenv("CELER_ENABLE_PROFILING").empty())
        {
            CELER_LOG(info) << "Enabling profiling support since the "
                               "'CELER_ENABLE_PROFILING' "
                               "environment variable is present and non-empty";
            return true;
        }
        return false;
    }();
    return result;
}

void ScopedProfiling::activate([[maybe_unused]] Input const& input) noexcept
{
    TRACE_EVENT_BEGIN("Celeritas",
                      perfetto::DynamicString{std::string{input.name}});
}
void ScopedProfiling::deactivate() noexcept
{
    TRACE_EVENT_END("Celeritas");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas