//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/StackAllocatorData.hh"
#include "geometry/GeoData.hh"
#include "geometry/GeoMaterialData.hh"
#include "physics/base/CutoffData.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleData.hh"
#include "physics/base/PhysicsData.hh"
#include "physics/base/Secondary.hh"
#include "physics/em/AtomicRelaxationData.hh"
#include "physics/material/MaterialData.hh"
#include "random/RngData.hh"

#include "SimData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Parameters for controlling state sizes etc.
 */
struct ControlOptions
{
    real_type secondary_stack_factor = 3; // Secondary storage per state size

    //! True if all options are valid
    explicit CELER_FUNCTION operator bool() const
    {
        return secondary_stack_factor > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Immutable problem data.
 */
template<Ownership W, MemSpace M>
struct CoreParamsData
{
    GeoParamsData<W, M>         geometry;
    GeoMaterialParamsData<W, M> geo_mats;
    MaterialParamsData<W, M>    materials;
    ParticleParamsData<W, M>    particles;
    CutoffParamsData<W, M>      cutoffs;
    PhysicsParamsData<W, M>     physics;
    AtomicRelaxParamsData<W, M> relaxation; // TODO: move into physics
    RngParamsData<W, M>         rng;

    ControlOptions control;

    //! True if all params are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && geo_mats && materials && particles && cutoffs
               && physics && control;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CoreParamsData& operator=(const CoreParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry   = other.geometry;
        geo_mats   = other.geo_mats;
        materials  = other.materials;
        particles  = other.particles;
        cutoffs    = other.cutoffs;
        physics    = other.physics;
        relaxation = other.relaxation;
        rng        = other.rng;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Thread-local state data.
 */
template<Ownership W, MemSpace M>
struct CoreStateData
{
    template<class T>
    using Items = StateCollection<T, W, M>;

    GeoStateData<W, M>         geometry;
    MaterialStateData<W, M>    materials;
    ParticleStateData<W, M>    particles;
    PhysicsStateData<W, M>     physics;
    AtomicRelaxStateData<W, M> relaxation;
    RngStateData<W, M>         rng;
    SimStateData<W, M>         sim;

    // Stacks
    StackAllocatorData<Secondary, W, M> secondaries;

    // Raw data
    Items<real_type>   step_length;       // TODO: step limiter
    Items<real_type>   energy_deposition; // TODO: move to physics?
    Items<Interaction> interactions;      // TODO: to be removed

    //! Number of state elements
    CELER_FUNCTION size_type size() const { return particles.size(); }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && materials && particles && physics && rng && sim
               && secondaries && !step_length.empty()
               && !energy_deposition.empty() && !interactions.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CoreStateData& operator=(CoreStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry          = other.geometry;
        materials         = other.materials;
        particles         = other.particles;
        physics           = other.physics;
        rng               = other.rng;
        relaxation        = other.relaxation;
        sim               = other.sim;
        secondaries       = other.secondaries;
        step_length       = other.step_length;
        energy_deposition = other.energy_deposition;
        interactions      = other.interactions;
        return *this;
    }
};

using ParamsDeviceRef
    = CoreParamsData<Ownership::const_reference, MemSpace::device>;
using ParamsHostRef
    = CoreParamsData<Ownership::const_reference, MemSpace::host>;
using StateDeviceRef = CoreStateData<Ownership::reference, MemSpace::device>;
using StateHostRef   = CoreStateData<Ownership::reference, MemSpace::host>;

//---------------------------------------------------------------------------//
/*!
 * Resize states in host code.
 */
template<MemSpace M>
inline void
resize(CoreStateData<Ownership::value, M>*                               data,
       const CoreParamsData<Ownership::const_reference, MemSpace::host>& params,
       size_type                                                         size)
{
    CELER_EXPECT(data);
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);
    resize(&data->geometry, params.geometry, size);
    resize(&data->materials, params.materials, size);
    resize(&data->particles, params.particles, size);
    resize(&data->physics, params.physics, size);
    resize(&data->relaxation, params.relaxation, size);
    resize(&data->rng, params.rng, size);
    resize(&data->sim, size);

    auto sec_size
        = static_cast<size_type>(size * params.control.secondary_stack_factor);
    resize(&data->secondaries, sec_size);

    resize(&data->step_length, size);
    resize(&data->energy_deposition, size);

    // Initialize empty interactions
    StateCollection<Interaction, Ownership::value, MemSpace::host> interactions;
    std::vector<Interaction> initial_state(size, Interaction{});
    make_builder(&interactions)
        .insert_back(initial_state.begin(), initial_state.end());
    data->interactions = interactions;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
