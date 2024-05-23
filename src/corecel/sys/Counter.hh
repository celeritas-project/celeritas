//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Counter.hh
//! \brief Numeric tracing counter
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>
#include <type_traits>

#include "celeritas_config.h"

namespace celeritas
{
template<class T>
class Counter
{
    static_assert(std::is_arithmetic_v<T>, "Only support numeric counters");

  public:
    Counter(std::string_view, T);
};

#if !CELERITAS_USE_PERFETTO
template<class T>
Counter<T>::Counter(std::string_view, T)
{
}
#endif

}  // namespace celeritas
