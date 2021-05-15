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
#include "sim/SimInterface.hh"

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

    real_type secondary_stack_factor = 3; // Secondary storage per state size

    //! True if all options are valid
    explicit operator bool() const { return secondary_stack_factor > 0; }
};

//---------------------------------------------------------------------------//
// TODO: unify with sim/TrackInterface
template<Ownership W, MemSpace M>
struct ParamsData
{
    celeritas::GeoParamsData<W, M>         geometry;
    celeritas::GeoMaterialParamsData<W, M> geo_mats;
    celeritas::MaterialParamsData<W, M>    materials;
    celeritas::ParticleParamsData<W, M>    particles;
    celeritas::CutoffParamsData<W, M>      cutoffs;
    celeritas::PhysicsParamsData<W, M>     physics;
    celeritas::RngParamsData<W, M>         rng;

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
        rng       = other.rng;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// TODO: unify with sim/TrackInterface
template<Ownership W, MemSpace M>
struct StateData
{
    template<class T>
    using Items     = celeritas::StateCollection<T, W, M>;
    using real_type = celeritas::real_type;

    celeritas::GeoStateData<W, M>      geometry;
    celeritas::MaterialStateData<W, M> materials;
    celeritas::ParticleStateData<W, M> particles;
    celeritas::PhysicsStateData<W, M>  physics;
    celeritas::RngStateData<W, M>      rng;
    celeritas::SimStateData<W, M>      sim;

    // Stacks
    celeritas::StackAllocatorData<celeritas::Secondary, W, M> secondaries;

    // Raw data
    Items<real_type>              step_length;
    Items<real_type>              energy_deposition;
    Items<celeritas::Interaction> interactions;

    //! Number of state elements
    CELER_FUNCTION celeritas::size_type size() const
    {
        return particles.size();
    }

    //! True if all params are assigned
    explicit operator bool() const
    {
        return geometry && materials && particles && physics && rng && sim
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
        sim               = other.sim;
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
    resize(&data->geometry, params.geometry, size);
    resize(&data->materials, params.materials, size);
    resize(&data->particles, params.particles, size);
    resize(&data->physics, params.physics, size);
    resize(&data->rng, params.rng, size);
    resize(&data->sim, size);

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
