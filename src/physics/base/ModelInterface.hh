//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ModelInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "base/StackAllocator.hh"
#include "base/Types.hh"
#include "random/cuda/RngInterface.hh"
#include "physics/material/MaterialInterface.hh"
#include "Secondary.hh"
#include "ParticleInterface.hh"
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
    ParticleParamsData<Ownership::const_reference, MemSpace::device> particle;
    MaterialParamsData<Ownership::const_reference, MemSpace::device> material;
    PhysicsParamsData<Ownership::const_reference, MemSpace::device>  physics;

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
 * \todo Template on memory space, use Collection for direction (?).
 */
struct ModelInteractState
{
    ParticleStateData<Ownership::reference, MemSpace::device> particle;
    MaterialStateData<Ownership::reference, MemSpace::device> material;
    PhysicsStateData<Ownership::reference, MemSpace::device>  physics;
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
        CELER_EXPECT(*this);
        return particle.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Input and output device data to a generic Model::interact call.
 *
 * \todo Template on memory space, use Collection for interaction result.
 */
struct ModelInteractPointers
{
    using SecondaryAllocatorData
        = StackAllocatorData<Secondary, Ownership::reference, MemSpace::device>;

    ModelInteractParams    params;
    ModelInteractState     states;
    SecondaryAllocatorData secondaries;
    Span<Interaction>      result;

    //! True if valid
    CELER_FUNCTION operator bool() const
    {
        return params && states && secondaries
               && result.size() == states.size();
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
