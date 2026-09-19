//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/detail/LarRunnerDiagnostics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "celeritas/phys/GeneratorCounters.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Timing results for a single event.
 */
struct LarTiming
{
    using MapStrDouble = std::unordered_map<std::string, double>;

    double setup{};  //!< One-time initialization cost
    double run{};  //!< Total transport time (no initialization)
    double teardown{};  //!< Time to convert hits/btrs to lardataobj
    MapStrDouble actions{};  //!< Accumulated action times
    std::vector<double> steps{};  //!< Step times
};

//---------------------------------------------------------------------------//
/*!
 * Results from transporting all tracks in an event.
 */
struct LarRunnerDiagnostics
{
    LarTiming time{};
    CounterAccumStats counters{};
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
