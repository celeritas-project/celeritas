//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalSizes.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Per-process optical state and buffer capacities.
 *
 * This stores the resolved values from \c inp::OpticalStateCapacity: all
 * values have been validated and device-dependent defaults have been applied
 * at setup time. These should be *integrated* across streams, *per process*.
 */
struct OpticalSizes
{
    size_type primaries{};
    size_type tracks{};
    size_type generators{};

    size_type streams{};

    //! True if all values are assigned and valid
    explicit operator bool() const
    {
        return primaries && tracks && generators && streams;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
