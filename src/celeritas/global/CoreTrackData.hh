//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreTrackData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/StackAllocatorData.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoMaterialData.hh"
#include "celeritas/phys/CutoffData.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/PhysicsData.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/em/data/AtomicRelaxationData.hh"
#include "celeritas/mat/MaterialData.hh"
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
    real_type secondary_stack_factor = 3; //!< Secondary storage per state size
    ActionId  boundary_action;

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return secondary_stack_factor > 0 && boundary_action;
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
    AtomicRelaxParamsData<W, M> relaxation; // TODO: move into physics?
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
        geometry   = other.geometry;
        geo_mats   = other.geo_mats;
        materials  = other.materials;
        particles  = other.particles;
        cutoffs    = other.cutoffs;
        physics    = other.physics;
        relaxation = other.relaxation;
        rng        = other.rng;
        scalars    = other.scalars;
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

    // TODO: move to physics?
    AtomicRelaxStateData<W, M> relaxation;

    // Stacks
    StackAllocatorData<Secondary, W, M> secondaries;

    //! Number of state elements
    CELER_FUNCTION size_type size() const { return particles.size(); }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && materials && particles && physics && rng && sim
               && secondaries;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CoreStateData& operator=(CoreStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry    = other.geometry;
        materials   = other.materials;
        particles   = other.particles;
        physics     = other.physics;
        rng         = other.rng;
        sim         = other.sim;
        relaxation  = other.relaxation;
        secondaries = other.secondaries;
        return *this;
    }
};

using CoreParamsDeviceRef
    = CoreParamsData<Ownership::const_reference, MemSpace::device>;
using CoreParamsHostRef
    = CoreParamsData<Ownership::const_reference, MemSpace::host>;
using CoreStateDeviceRef
    = CoreStateData<Ownership::reference, MemSpace::device>;
using CoreStateHostRef = CoreStateData<Ownership::reference, MemSpace::host>;

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

using CoreHostRef   = CoreRef<MemSpace::host>;
using CoreDeviceRef = CoreRef<MemSpace::device>;

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
        = static_cast<size_type>(size * params.scalars.secondary_stack_factor);
    resize(&data->secondaries, sec_size);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
