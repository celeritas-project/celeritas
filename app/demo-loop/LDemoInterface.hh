//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/StackAllocatorInterface.hh"
#include "geometry/GeoInterface.hh"
#include "geometry/GeoMaterialInterface.hh"
#include "physics/base/CutoffInterface.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleInterface.hh"
#include "physics/base/PhysicsInterface.hh"
#include "physics/base/Secondary.hh"
#include "physics/material/MaterialInterface.hh"
#include "random/RngInterface.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
using celeritas::MemSpace;
using celeritas::Ownership;

//---------------------------------------------------------------------------//
/*!
 * Parameters for controlling state sizes etc.
 */
struct ControlOptions
{
    using real_type = celeritas::real_type;

    unsigned int rng_seed            = 20210318u;
    real_type secondary_stack_factor = 3; // Secondary storage per state size

    //! True if all options are valid
    explicit operator bool() const { return secondary_stack_factor > 0; }
};

//---------------------------------------------------------------------------//
// TODO: unify with sim/TrackInterface
template<Ownership W, MemSpace M>
struct ParamsData
{
    celeritas::GeoParamsPointers           geometry;
    celeritas::GeoMaterialParamsData<W, M> geo_mats;

    celeritas::MaterialParamsData<W, M> materials;
    celeritas::ParticleParamsData<W, M> particles;
    celeritas::CutoffParamsData<W, M>   cutoffs;
    celeritas::PhysicsParamsData<W, M>  physics;

    ControlOptions control;

    //! True if all params are assigned
    explicit operator bool() const
    {
        return geometry && geo_mats && materials && particles && cutoffs
               && physics && control;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ParamsData& operator=(const ParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry  = other.geometry;
        geo_mats  = other.geo_mats;
        materials = other.materials;
        particles = other.particles;
        cutoffs   = other.cutoffs;
        physics   = other.physics;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// TODO: unify with sim/TrackInterface
template<Ownership W, MemSpace M>
struct StateData
{
    using real_type = celeritas::real_type;

    // TODO: geometry state (this is a placeholdder)
    int geometry = 0;

    celeritas::MaterialStateData<W, M> materials;
    celeritas::ParticleStateData<W, M> particles;
    celeritas::PhysicsStateData<W, M>  physics;
    celeritas::RngStateData<W, M>      rng;

    // Stacks
    celeritas::StackAllocatorData<celeritas::Secondary, W, M> secondaries;

    // Raw data
    celeritas::Collection<real_type, W, M>              step_length;
    celeritas::Collection<real_type, W, M>              energy_deposition;
    celeritas::Collection<celeritas::Interaction, W, M> interactions;

    //! Number of state elements
    CELER_FUNCTION celeritas::size_type size() const
    {
        return particles.size();
    }

    //! True if all params are assigned
    explicit operator bool() const
    {
        return geometry && materials && particles && physics && rng
               && secondaries && !step_length.empty()
               && !energy_deposition.empty() && !interactions.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    StateData& operator=(StateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry          = other.geometry;
        materials         = other.materials;
        particles         = other.particles;
        physics           = other.physics;
        rng               = other.rng;
        secondaries       = other.secondaries;
        step_length       = other.step_length;
        energy_deposition = other.energy_deposition;
        interactions      = other.interactions;
        return *this;
    }
};

using ParamsDeviceRef
    = ParamsData<Ownership::const_reference, MemSpace::device>;
using StateDeviceRef = StateData<Ownership::reference, MemSpace::device>;

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
/*!
 * Resize particles states in host code.
 */
template<MemSpace M>
inline void
resize(StateData<Ownership::value, M>*                               data,
       const ParamsData<Ownership::const_reference, MemSpace::host>& params,
       celeritas::size_type                                          size)
{
    CELER_EXPECT(data);
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);
    resize(&data->materials, params.materials, size);
    resize(&data->particles, params.particles, size);
    resize(&data->physics, params.physics, size);
    CELER_NOT_IMPLEMENTED("TODO: resize remaining states");

    auto sec_size = static_cast<celeritas::size_type>(
        size * params.control.secondary_stack_factor);
    resize(&data->secondaries, sec_size);

    resize(&data->step_length, size);
    resize(&data->energy_deposition, size);
    resize(&data->interactions, size);
}
#endif

//---------------------------------------------------------------------------//
} // namespace demo_loop
