//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Counter.hh
//! \brief Numeric tracing counter
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Macros.hh"

namespace celeritas
{
#if CELERITAS_USE_PERFETTO
//---------------------------------------------------------------------------//
// Simple tracing counter
template<class T>
CELER_FUNCTION void trace_counter(char const* name, T value);
#else
//! No tracing backend - noop
template<class T>
CELER_FUNCTION inline void trace_counter(char const*, T)
{
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
