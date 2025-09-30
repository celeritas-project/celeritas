//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/TraceCounter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "corecel/Config.hh"

#include "ScopedProfiling.hh"

#include "detail/TraceCounterImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simple performance tracing counter.
 * \tparam T Arithmetic counter type
 *
 * Records a named value at the current timestamp which
 * can then be displayed on a timeline. This is implemented for Perfetto and
 * CUDA NVTX.
 *
 * See https://perfetto.dev/docs/instrumentation/track-events#counters
 */
template<class T>
inline void trace_counter(char const* name, T value)
{
    static_assert(std::is_arithmetic_v<T>,
                  "Only numeric counters are supported");
    if ((CELERITAS_USE_PERFETTO || CELERITAS_USE_CUDA)
        && ScopedProfiling::enabled())
    {
        using counter_type = detail::trace_counter_type<T>;
        detail::trace_counter_impl<counter_type>(name, value);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
