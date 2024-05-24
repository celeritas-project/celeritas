//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Counter.hh
//! \brief Numeric tracing counter
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Macros.hh"

namespace celeritas
{

#if CELERITAS_USE_PERFETTO
/*!
 * Simple tracing counter. Records a named value at the current timestamp which
 * can then be displayed on a timeline. Only supported on host, this compiles
 * but is a noop on device.
 * See https://perfetto.dev/docs/instrumentation/track-events#counters
 * @tparam T Arithmetic counter type
 */
template<class T>
CELER_FUNCTION void trace_counter(char const* name, T value);
#else
// noop
template<class T>
CELER_FUNCTION inline void trace_counter(char const*, T)
{
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
