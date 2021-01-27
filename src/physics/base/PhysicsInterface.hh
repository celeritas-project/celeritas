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
// PARAMS
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
 */
struct PhysicsTrackState
{
    ModelId model_id; //!< Selected model if interacting
};

//---------------------------------------------------------------------------//
// STATE
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
