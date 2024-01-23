//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/CoreStateCounters.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Counters for track initialization and activity.
 *
 * These counters are updated *by value on the host at every step* so they
 * should not be stored in TrackInitStateData because then the device-memory
 * copy will not be synchronized.
 */
struct CoreStateCounters
{
    // Initialization input
    size_type num_vacancies{};  //!< Number of unused track slots
    size_type num_primaries{};  //!< Number of primaries to be converted
    size_type num_initializers{};  //!< Number of track initializers

    // Diagnostic output
    size_type num_secondaries{};  //!< Number of secondaries produced in a step
    size_type num_active{};  //!< Number of active tracks at start of a step
    size_type num_alive{};  //!< Number of alive tracks at end of step
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
