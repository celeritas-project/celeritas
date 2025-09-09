//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.cc
//! \brief The perfetto implementation of \c ScopedProfiling
//---------------------------------------------------------------------------//
#include "ScopedProfiling.hh"

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"

#include "Environment.hh"

#if CELERITAS_USE_PERFETTO
#    include <perfetto.h>

#    include "detail/TrackEvent.perfetto.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Whether profiling is enabled.
 *
 * This is true only if the \c CELER_ENABLE_PROFILING environment variable is
 * set to a non-empty value. Profiling is never enabled if CUDA/HIP/Perfetto
 * isn't available.
 */
bool use_profiling()
{
    static bool const result = [] {
        if (!celeritas::getenv("CELER_ENABLE_PROFILING").empty())
        {
            if constexpr (CELERITAS_USE_PERFETTO || CELER_USE_DEVICE)
            {
                if constexpr (CELERITAS_USE_HIP && !CELERITAS_HAVE_ROCTX)
                {
                    CELER_LOG(warning) << "Disabling profiling support "
                                          "since ROC-TX is unavailable";
                    return false;
                }
                CELER_LOG(info)
                    << "Enabling profiling support since the "
                       "'CELER_ENABLE_PROFILING' "
                       "environment variable is present and non-empty";
                return true;
            }
            CELER_LOG(warning)
                << "CELER_ENABLE_PROFILING is set but Celeritas "
                   "was compiled without a profiling backend.";
        }
        return false;
    }();
    return result;
}

#if CELERITAS_USE_PERFETTO
//---------------------------------------------------------------------------//
/*!
 * Start a thread-local slice track event
 */
void ScopedProfiling::activate(Input const& input) noexcept
{
    TRACE_EVENT_BEGIN(detail::perfetto_track_event_category,
                      perfetto::DynamicString{std::string{input.name}});
}

//---------------------------------------------------------------------------//
/*!
 * End the slice track event that was started on the current thread
 */
void ScopedProfiling::deactivate() noexcept
{
    TRACE_EVENT_END(detail::perfetto_track_event_category);
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
