//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Persistent shared physics data.
 */
struct PhysicsParamsPointers
{
    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return true; }
};

//---------------------------------------------------------------------------//
/*!
 * Physics state data for a single track.
 *
 * - Remaining number of mean free paths to the next discrete interaction
 * - Maximum step length (limited by range, energy loss, and interaction)
 * - Selected model ID if undergoing an interaction
 */
struct PhysicsTrackState
{
    ModelId model_id; //!< Selected model if interacting
};

//---------------------------------------------------------------------------//
/*!
 * Dynamic physics (models, processes) state data.
 */
struct PhysicsStatePointers
{
    Span<PhysicsTrackState> state; //!< Track state [track]

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !state.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return state.size(); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
