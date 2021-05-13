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
#include "random/RngInterface.hh"
#include "SimInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Immutable problem data.
 *
 * TODO: unify TrackInterface with the demo-loop LDemoInterface (of which this
 * is currently a subset).
 */
template<Ownership W, MemSpace M>
struct ParamsData
{
    GeoParamsData<W, M>      geometry;
    MaterialParamsData<W, M> materials;
    ParticleParamsData<W, M> particles;
    RngParamsData<W, M>      rng;

    //! True if all params are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && materials && particles;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ParamsData& operator=(const ParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry  = other.geometry;
        materials = other.materials;
        particles = other.particles;
        rng       = other.rng;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Thread-local state data.
 */
template<Ownership W, MemSpace M>
struct StateData
{
    template<class T>
    using Items = StateCollection<T, W, M>;

    GeoStateData<W, M>      geometry;
    ParticleStateData<W, M> particles;
    RngStateData<W, M>      rng;
    SimStateData<W, M>      sim;

    // Raw data
    Items<celeritas::Interaction> interactions;

    //! Number of tracks
    CELER_FUNCTION celeritas::size_type size() const
    {
        return particles.size();
    }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && particles && rng && sim && !interactions.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    StateData& operator=(StateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry     = other.geometry;
        particles    = other.particles;
        rng          = other.rng;
        sim          = other.sim;
        interactions = other.interactions;
        return *this;
    }
};

using ParamsDeviceRef
    = ParamsData<Ownership::const_reference, MemSpace::device>;
using StateDeviceRef = StateData<Ownership::reference, MemSpace::device>;

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
/*!
 * Resize states in host code.
 */
template<MemSpace M>
inline void
resize(StateData<Ownership::value, M>*                               data,
       const ParamsData<Ownership::const_reference, MemSpace::host>& params,
       size_type                                                     size)
{
    CELER_EXPECT(data);
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);

    resize(&data->geometry, params.geometry, size);
    resize(&data->particles, params.particles, size);
    resize(&data->rng, params.rng, size);
    resize(&data->sim, size);
    resize(&data->interactions, size);
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
