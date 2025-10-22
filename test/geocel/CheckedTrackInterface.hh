//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/CheckedTrackInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Interface for accessing checking/instrumentation data from track views.
 *
 * This interface provides access to diagnostic counters and utilities for
 * track views that perform validation and checking during navigation.
 */
class CheckedTrackInterface
{
  public:
    //! Virtual destructor for polymorphic use
    virtual ~CheckedTrackInterface() = default;

    //! Get the number of calls to find_next_step
    virtual size_type intersect_count() const = 0;

    //! Get the number of calls to find_safety
    virtual size_type safety_count() const = 0;

    //! Reset the intersection and safety counters
    virtual void reset_count() = 0;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
