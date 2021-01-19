//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StatePointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"
#include "geometry/GeoStatePointers.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleStatePointers.hh"
#include "random/cuda/RngStatePointers.hh"
#include "SimStatePointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * View to the track state data
 */
struct StatePointers
{
    ParticleStatePointers particle;
    GeoStatePointers      geo;
    SimStatePointers      sim;
    RngStatePointers      rng;
    Span<Interaction>     interactions;

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
