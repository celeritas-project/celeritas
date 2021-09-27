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
#include "random/RngInterface.hh"
#include "physics/material/MaterialInterface.hh"
#include "physics/base/CutoffInterface.hh"
#include "physics/em/AtomicRelaxationInterface.hh"
#include "sim/SimInterface.hh"
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
template<MemSpace M>
struct ModelInteractParamsRefs
{
    //// TYPES ////

    template<template<Ownership, MemSpace> class P>
    using ParamsCRef = P<Ownership::const_reference, M>;

    //// DATA ////

    ParamsCRef<ParticleParamsData>    particle;
    ParamsCRef<MaterialParamsData>    material;
    ParamsCRef<PhysicsParamsData>     physics;
    ParamsCRef<CutoffParamsData>      cutoffs;
    ParamsCRef<AtomicRelaxParamsData> relaxation;

    //// METHODS ////

    //! True if assigned
    CELER_FUNCTION operator bool() const
    {
        return physics && particle && material && cutoffs;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Mutable state data needed when interacting with a model.
 *
 * \todo The use of a StateItems<Real3> violates encapsulation; ideally we
 * could use a GeoStateData but then we'd have to include vecgeom...
 */
template<MemSpace M>
struct ModelInteractStateRefs
{
    //// TYPES ////

    template<template<Ownership, MemSpace> class S>
    using StateRef = S<Ownership::reference, M>;
    template<class T>
    using StateItems = celeritas::StateCollection<T, Ownership::reference, M>;
    template<class T>
    using AllocatorRef = StackAllocatorData<T, Ownership::reference, M>;

    //// DATA ////

    StateRef<ParticleStateData>    particle;
    StateRef<MaterialStateData>    material;
    StateRef<PhysicsStateData>     physics;
    StateRef<RngStateData>         rng;
    StateRef<SimStateData>         sim;
    StateRef<AtomicRelaxStateData> relaxation;

    StateItems<Real3>       direction;
    StateItems<Interaction> interactions;

    AllocatorRef<Secondary> secondaries;

    //// METHODS ////

    //! True if assigned
    CELER_FUNCTION operator bool() const
    {
        // clang-format off
        return particle
               && material.size() == particle.size()
               && physics.size() == particle.size()
               && direction.size() == particle.size()
               && rng.size() == particle.size()
               && sim.size() == particle.size()
               && interactions.size() == particle.size()
               && secondaries;
        // clang-format on
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
 * All data needed to interact with a model.
 */
template<MemSpace M>
struct ModelInteractRefs
{
    ModelInteractParamsRefs<M> params;
    ModelInteractStateRefs<M>  states;

    //! True if assigned
    CELER_FUNCTION operator bool() const { return params && states; }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
