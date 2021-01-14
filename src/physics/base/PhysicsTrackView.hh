//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "PhysicsInterface.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/material/Types.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Physics data for a track.
 *
 * The physics track view provides an interface for data and operations
 * common to most processes and models.
 */
class PhysicsTrackView
{
  public:
    //!@{
    //! Type aliases
    //!@}

  public:
    // Construct from "dynamic" state and "static" particle definitions
    inline CELER_FUNCTION PhysicsTrackView(const PhysicsParamsPointers& params,
                                           const PhysicsStatePointers&  states,
                                           ParticleDefId particle,
                                           MaterialDefId material,
                                           ThreadId      id);

    // Select an interaction model (or {} for no interaction)
    inline CELER_FUNCTION void model_id(ModelId);

    //// DYNAMIC PROPERTIES (pure accessors, free) ////

    // Selected model if interacting
    CELER_FORCEINLINE_FUNCTION ModelId model_id() const;

  private:
    const PhysicsParamsPointers& params_;
    const PhysicsStatePointers&  states_;
    const ThreadId               tid_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "PhysicsTrackView.i.hh"
