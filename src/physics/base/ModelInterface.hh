//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ModelInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "base/Types.hh"
#include "random/cuda/RngStatePointers.hh"
#include "physics/material/MaterialParamsPointers.hh"
#include "physics/material/MaterialStatePointers.hh"
#include "SecondaryAllocatorPointers.hh"
#include "ParticleParamsPointers.hh"
#include "ParticleStatePointers.hh"
#include "Interaction.hh"
#include "PhysicsInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared parameters needed when interacting with a model.
 */
struct ModelInteractParams
{
    ParticleParamsPointers particle;
    MaterialParamsPointers material;
    PhysicsParamsPointers  physics;

    //! True if valid
    CELER_FUNCTION operator bool() const
    {
        return physics && particle && material;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Per-track states needed when interacting with a model.
 *
 * \todo The use of a Span<Real3> violates encapsulation; ideally we could use
 * a GeoStatePointers or directly pass the geo state store.
 */
struct ModelInteractState
{
    ParticleStatePointers particle;
    MaterialStatePointers material;
    PhysicsStatePointers  physics;
    Span<const Real3>     direction;
    RngStatePointers      rng;

    //! True if valid
    CELER_FUNCTION operator bool() const
    {
        return particle && material && physics && !direction.empty() && rng;
    }

    //! Number of particle tracks
    CELER_FUNCTION size_type size() const
    {
        REQUIRE(*this);
        return particle.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Input and output device data to a generic Model::interact call.
 */
struct ModelInteractPointers
{
    ModelInteractParams        params;
    ModelInteractState         states;
    SecondaryAllocatorPointers secondaries;
    Span<Interaction>          result;

    //! True if valid
    CELER_FUNCTION operator bool() const
    {
        return params && states && secondaries && !result.empty();
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
