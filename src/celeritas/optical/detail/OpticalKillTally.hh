//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalKillTally.hh
//! \brief Host-only debug tally of optical photon kills by site and volume.
//!
//! Enabled with CELER_DEBUG_OPTICAL_FATES=1; counts are printed to stderr at
//! process exit. CPU-only diagnostic: calls must be guarded with
//! !CELER_DEVICE_COMPILE at the call site.
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"

#if !CELER_DEVICE_COMPILE
#    include <atomic>
#    include <cstdio>
#    include <cstdlib>
#    include <map>
#    include <mutex>
#    include <string>

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
inline void
tally_optical_kill(char const* site, unsigned int volume, bool ultraviolet)
{
    static bool const enabled
        = std::getenv("CELER_DEBUG_OPTICAL_FATES") != nullptr;
    if (!enabled)
    {
        return;
    }

    struct Tally
    {
        std::mutex mutex;
        std::map<std::string, long> counts;
        ~Tally()
        {
            for (auto const& kv : counts)
            {
                std::fprintf(stderr,
                             "[OPTICAL-FATES] %s %ld\n",
                             kv.first.c_str(),
                             kv.second);
            }
        }
    };
    static Tally tally;

    char buf[128];
    std::snprintf(buf,
                  sizeof(buf),
                  "%s vol=%u %s",
                  site,
                  volume,
                  ultraviolet ? "uv" : "vis");
    std::lock_guard<std::mutex> lock(tally.mutex);
    ++tally.counts[buf];
}


//---------------------------------------------------------------------------//
//! Latched track slot for single-photon surface tracing
//! (CELER_DEBUG_SURFACE_TRACE)
inline std::atomic<int>& traced_slot()
{
    static std::atomic<int> slot{-1};
    return slot;
}

inline bool surface_trace_enabled()
{
    static bool const enabled
        = std::getenv("CELER_DEBUG_SURFACE_TRACE") != nullptr;
    return enabled;
}

inline void trace_surface(char const* msg)
{
    static std::atomic<int> budget{500};
    if (budget.fetch_sub(1) <= 0)
    {
        return;
    }
    std::fprintf(stderr, "[SURFTRACE] %s\n", msg);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
#endif
