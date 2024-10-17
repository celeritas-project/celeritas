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
 * Counters for within-step track initialization and activity.
 *
 * These counters are updated *by value on the host at every step* so they
 * should not be stored in TrackInitStateData because then the device-memory
 * copy will not be synchronized.
 *
 * \todo Drop the 'num' prefix since we know they're counters.
 */
struct CoreStateCounters
{
    //!@{
    //! \name Set after primaries are generated
    size_type num_generated{0};  //!< Number of track initializers created
    //!@}
    //
    //!@{
    //! \name Updated during generation and initialization
    size_type num_initializers{0};  //!< Number of track initializers
    size_type num_vacancies{0};  //!< Number of empty track slots
    //!@}

    //!@{
    //! \name Set after tracks are initialized
    size_type num_active{0};  //!< Number of active tracks at start
    //!@}

    //!@{
    //! \name Set after secondaries are generated
    size_type num_secondaries{0};  //!< Number of secondaries produced
    size_type num_alive{0};  //!< Number of alive tracks at end
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
