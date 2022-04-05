//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//! Type definitions for simulation management
//---------------------------------------------------------------------------//
#pragma once

#include "base/OpaqueId.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Counter for the initiating event for a track
using EventId = OpaqueId<struct Event>;
//! Unique ID (for an event) of a track among all primaries and secondaries
using TrackId = OpaqueId<struct Track>;
//! End-of-step (or perhaps someday within-step?) action to take
using ActionId = OpaqueId<class ActionInterface>;

//---------------------------------------------------------------------------//
//! Step length and limiting action to take
struct StepLimit
{
    real_type step{};
    ActionId  action{};

    //! Whether a step limit has been determined
    explicit CELER_FUNCTION operator bool() const
    {
        CELER_ASSERT(step >= 0);
        return static_cast<bool>(action);
    }
};

//---------------------------------------------------------------------------//
//! Whether a track slot is alive, inactive, or dying
enum class TrackStatus : signed char
{
    killed   = -1, //!< Killed inside the step, awaiting replacement
    inactive = 0,  //!< No tracking in this thread slot
    alive    = 1   //!< Track is active and alive
};

//---------------------------------------------------------------------------//
} // namespace celeritas
