//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreSizes.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Per-process core state and buffer capacities.
 *
 * This stores the resolved values from \c inp::CoreStateCapacity: all values
 * have been validated and device-dependent defaults have been applied at setup
 * time. These should be *integrated* across streams, *per process*.
 */
struct CoreSizes
{
    size_type primaries{};
    size_type tracks{};
    size_type initializers{};
    size_type secondaries{};
    size_type events{};

    size_type streams{};
    size_type processes{};

    //! True if all values are assigned and valid
    explicit operator bool() const
    {
        return primaries && tracks && initializers && secondaries && events
               && streams && processes;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
