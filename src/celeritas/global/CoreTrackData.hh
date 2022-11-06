//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreTrackData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoMaterialData.hh"
#include "celeritas/mat/MaterialData.hh"
#include "celeritas/phys/CutoffData.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/PhysicsData.hh"
#include "celeritas/random/RngData.hh"
#include "celeritas/track/SimData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Memspace-independent core variables.
 */
struct CoreScalars
{
    ActionId boundary_action;
    ActionId propagation_limit_action;

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return boundary_action && propagation_limit_action;
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
    RngParamsData<W, M>         rng;

    CoreScalars scalars;

    //! True if all params are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && geo_mats && materials && particles && cutoffs
               && physics && scalars;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CoreParamsData& operator=(const CoreParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry  = other.geometry;
        geo_mats  = other.geo_mats;
        materials = other.materials;
        particles = other.particles;
        cutoffs   = other.cutoffs;
        physics   = other.physics;
        rng       = other.rng;
        scalars   = other.scalars;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Thread-local state data.
 *
 * TODO: standardize variable names
 */
template<Ownership W, MemSpace M>
struct CoreStateData
{
    template<class T>
    using Items = StateCollection<T, W, M>;

    GeoStateData<W, M>      geometry;
    MaterialStateData<W, M> materials;
    ParticleStateData<W, M> particles;
    PhysicsStateData<W, M>  physics;
    RngStateData<W, M>      rng;
    SimStateData<W, M>      sim;

    //! Number of state elements
    CELER_FUNCTION size_type size() const { return particles.size(); }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && materials && particles && physics && rng && sim;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CoreStateData& operator=(CoreStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry  = other.geometry;
        materials = other.materials;
        particles = other.particles;
        physics   = other.physics;
        rng       = other.rng;
        sim       = other.sim;
        return *this;
    }
};

// TODO: DEPRECATED TYPE ALIASES
using CoreParamsDeviceRef = DeviceCRef<CoreParamsData>;
using CoreParamsHostRef   = HostCRef<CoreParamsData>;
using CoreStateDeviceRef  = DeviceRef<CoreStateData>;
using CoreStateHostRef    = HostRef<CoreStateData>;

//---------------------------------------------------------------------------//
/*!
 * Reference to core parameters and states.
 *
 * This is passed via \c ExplicitActionInterface::execute to launch kernels.
 */
template<MemSpace M>
struct CoreRef
{
    CoreParamsData<Ownership::const_reference, M> params;
    CoreStateData<Ownership::reference, M>        states;

    //! True if assigned
    CELER_FUNCTION operator bool() const { return params && states; }
};

// TODO: DEPRECATED TYPE ALIASES
using CoreHostRef   = CoreRef<MemSpace::host>;
using CoreDeviceRef = CoreRef<MemSpace::device>;

//---------------------------------------------------------------------------//
/*!
 * Resize states in host code.
 */
template<MemSpace M>
inline void resize(CoreStateData<Ownership::value, M>* state,
                   const HostCRef<CoreParamsData>&     params,
                   size_type                           size)
{
    CELER_EXPECT(state);
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);
    resize(&state->geometry, params.geometry, size);
    resize(&state->materials, params.materials, size);
    resize(&state->particles, params.particles, size);
    resize(&state->physics, params.physics, size);
    resize(&state->rng, params.rng, size);
    resize(&state->sim, size);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
