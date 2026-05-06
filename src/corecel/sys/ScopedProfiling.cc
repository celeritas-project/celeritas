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
 * This defaults to ON if CUDA is enabled since (as of CUDA 12.0) there is no
 * measurable performance impact for NVTX hooks. It defaults to OFF otherwise.
 */
bool ScopedProfiling::enabled()
{
    static bool const enabled_ = [] {
#if CELERITAS_USE_CUDA
        constexpr auto get_default_profiling = ScopedProfiling::is_nvtx_enabled;
#else
        constexpr auto get_default_profiling = [] { return false; };
#endif
        auto result = celeritas::getenv_flag_lazy("CELER_ENABLE_PROFILING",
                                                  get_default_profiling);
        if (result.value)
        {
            if constexpr (CELERITAS_USE_HIP && !CELERITAS_HAVE_ROCTX)
            {
                CELER_LOG(error) << "Profiling support is disabled since "
                                    "ROC-TX is unavailable";
                return false;
            }
            else if constexpr (!CELER_USE_DEVICE && !CELERITAS_USE_PERFETTO)
            {
                CELER_LOG(error) << "CELER_ENABLE_PROFILING is set but "
                                    "Celeritas was compiled without a "
                                    "profiling backend: code will run but no "
                                    "profiling will be generated";
                return false;
            }
        }

        // Log level is 'debug' if user-specified, 'warning' if defaulted to
        // false but Perfetto was compiled, 'debug' otherwise
        auto msg = world_logger()(CELER_CODE_PROVENANCE,
                                  !result.defaulted ? LogLevel::debug
                                  : (CELERITAS_USE_PERFETTO && !result.value)
                                      ? LogLevel::warning
                                      : LogLevel::debug);

        msg << (result.value ? "En" : "Dis") << "abling "
            << (CELERITAS_USE_PERFETTO ? "Perfetto"
                : CELERITAS_HAVE_ROCTX ? "ROC-TX"
                : CELERITAS_USE_CUDA   ? "NVTX"
                                       : "unavailable")
            << " performance profiling";
        return result.value;
    }();
    return enabled_;
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
