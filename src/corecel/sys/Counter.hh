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
template<class T>
class Counter
{
    static_assert(std::is_arithmetic_v<T>, "Only support numeric counters");

  public:
#if CELERITAS_USE_PERFETTO
    CELER_FUNCTION Counter(char const*, T);
#else
    CELER_FUNCTION Counter(char const*, T) {}
#endif
};

}  // namespace celeritas
