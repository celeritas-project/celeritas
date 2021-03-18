//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "geometry/GeoInterface.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleInterface.hh"
#include "physics/material/MaterialInterface.hh"
#include "random/cuda/RngInterface.hh"
#include "SimInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Immutable problem data.
 */
struct ParamPointers
{
    GeoParamsPointers                                                geo;
    MaterialParamsData<Ownership::const_reference, MemSpace::device> material;
    ParticleParamsData<Ownership::const_reference, MemSpace::device> particle;

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geo && material && particle;
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Thread-local state data.
 */
struct StatePointers
{
    ParticleStateData<Ownership::reference, MemSpace::device> particle;
    GeoStatePointers                                          geo;
    SimStatePointers                                          sim;
    RngStatePointers                                          rng;
    Span<Interaction>                                         interactions;

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return particle && geo && sim && rng && !interactions.empty();
    }

    //! Number of tracks
    CELER_FUNCTION size_type size() const { return particle.size(); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
