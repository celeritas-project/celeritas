//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CoreTrackData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/data/Collection.hh"
#include "corecel/random/data/RngData.hh"
#include "geocel/SurfaceData.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoData.hh"

#include "CoreTrackDataFwd.hh"
#include "MaterialData.hh"
#include "ParticleData.hh"
#include "PhysicsData.hh"
#include "SimData.hh"
#include "TrackInitData.hh"
#include "Types.hh"
#include "gen/CherenkovData.hh"
#include "gen/ScintillationData.hh"
#include "surface/SurfacePhysicsData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Memspace-independent core variables.
 */
struct CoreScalars
{
    ActionId tracking_cut_action;

    StreamId::size_type max_streams{0};

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const { return max_streams > 0; }
};

//---------------------------------------------------------------------------//
/*!
 * Immutable problem data.
 */
template<Ownership W, MemSpace M>
struct CoreParamsData
{
    GeoParamsData<W, M> geometry;
    MaterialParamsData<W, M> material;
    PhysicsParamsData<W, M> physics;
    RngParamsData<W, M> rng;
    SimParamsData<W, M> sim;
    SurfaceParamsData<W, M> surface;
    SurfacePhysicsParamsData<W, M> surface_physics;
    CherenkovData<W, M> cherenkov;
    ScintillationData<W, M> scintillation;

    CoreScalars scalars;

    //! True if all params are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && material && physics && surface && surface_physics
               && rng && sim && scalars;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CoreParamsData& operator=(CoreParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        geometry = other.geometry;
        material = other.material;
        physics = other.physics;
        rng = other.rng;
        sim = other.sim;
        surface = other.surface;
        surface_physics = other.surface_physics;
        scalars = other.scalars;
        cherenkov = other.cherenkov;
        scintillation = other.scintillation;
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

    GeoStateData<W, M> geometry;
    // TODO: should we cache the material ID?
    ParticleStateData<W, M> particle;
    PhysicsStateData<W, M> physics;
    RngStateData<W, M> rng;
    SimStateData<W, M> sim;
    SurfacePhysicsStateData<W, M> surface_physics;
    TrackInitStateData<W, M> init;

    //! Unique identifier for "thread-local" data.
    StreamId stream_id;

    //! Number of state elements
    CELER_FUNCTION size_type size() const { return geometry.size(); }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && particle && physics && rng && sim && surface_physics
               && init && stream_id;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CoreStateData& operator=(CoreStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry = other.geometry;
        particle = other.particle;
        physics = other.physics;
        rng = other.rng;
        sim = other.sim;
        surface_physics = other.surface_physics;
        init = other.init;
        stream_id = other.stream_id;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize states in host code.
 *
 * Initialize threads to track slots mapping.
 * Resize core states using parameter data, stream ID, and track slots.
 */
template<MemSpace M>
void resize(CoreStateData<Ownership::value, M>* state,
            HostCRef<CoreParamsData> const& params,
            StreamId stream_id,
            size_type size);

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
