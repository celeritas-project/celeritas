//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Counter.hh
//! \brief Numeric tracing counter
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "celeritas_config.h"
#include "corecel/Macros.hh"

namespace celeritas
{
/*!
 * Simple tracing counter. Records a named value at the current timestamp which
 * can then be displayed on a timeline. Only supported on host, this compiles
 * but is a noop on device.
 * See https://perfetto.dev/docs/instrumentation/track-events#counters
 * @tparam T Arithmetic counter type
 */
template<class T>
class Counter
{
    static_assert(std::is_arithmetic_v<T>, "Only support numeric counters");

  public:
#if CELERITAS_USE_PERFETTO
    // Record value for the counter name
    CELER_FUNCTION Counter(char const* name, T value);
#else
    // noop
    CELER_FUNCTION Counter(char const*, T) {}
#endif
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
