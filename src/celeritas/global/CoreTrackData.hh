//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreTrackData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/random/data/RngData.hh"
#include "geocel/SurfaceData.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/WentzelOKVIData.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoMaterialData.hh"
#include "celeritas/mat/MaterialData.hh"
#include "celeritas/phys/CutoffData.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/PhysicsData.hh"
#include "celeritas/track/SimData.hh"
#include "celeritas/track/TrackInitData.hh"

#include "CoreTrackDataFwd.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreParams;

//---------------------------------------------------------------------------//
/*!
 * Memspace-independent core variables.
 */
struct CoreScalars
{
    ActionId boundary_action;
    ActionId propagation_limit_action;
    ActionId tracking_cut_action;  //!< Deposit a track's energy locally

    // TODO: this is a hack until we improve the along-step interface
    ActionId along_step_user_action;
    ActionId along_step_neutral_action;

    StreamId::size_type max_streams{0};

    // Non-owning pointer to core params ONLY for diagnostics:
    // see DebugIO.json.cc
    ObserverPtr<CoreParams const, MemSpace::host> host_core_params{nullptr};

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return boundary_action && propagation_limit_action
               && tracking_cut_action && along_step_user_action
               && along_step_neutral_action && max_streams > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Immutable problem data.
 */
template<Ownership W, MemSpace M>
struct CoreParamsData
{
    GeoParamsData<W, M> geometry;
    MaterialParamsData<W, M> materials;
    GeoMaterialParamsData<W, M> geo_mats;
    ParticleParamsData<W, M> particles;
    CutoffParamsData<W, M> cutoffs;
    PhysicsParamsData<W, M> physics;
    RngParamsData<W, M> rng;
    SimParamsData<W, M> sim;
    SurfaceParamsData<W, M> surface;
    TrackInitParamsData<W, M> init;
    WentzelOKVIData<W, M> wentzel;

    CoreScalars scalars;

    //! True if all params are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        // OPTIONAL: surface, volumes
        return geometry && materials && geo_mats && particles && cutoffs
               && physics && rng && sim && init && scalars;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CoreParamsData& operator=(CoreParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        geometry = other.geometry;
        materials = other.materials;
        geo_mats = other.geo_mats;
        particles = other.particles;
        cutoffs = other.cutoffs;
        physics = other.physics;
        rng = other.rng;
        sim = other.sim;
        surface = other.surface;
        init = other.init;
        wentzel = other.wentzel;
        scalars = other.scalars;
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
    using ThreadItems = Collection<T, W, M, ThreadId>;

    GeoStateData<W, M> geometry;
    MaterialStateData<W, M> materials;
    ParticleStateData<W, M> particles;
    PhysicsStateData<W, M> physics;
    RngStateData<W, M> rng;
    SimStateData<W, M> sim;
    TrackInitStateData<W, M> init;

    //! Indirection array for sorting (empty if unsorted)
    ThreadItems<TrackSlotId::size_type> track_slots;

    //! Unique identifier for "thread-local" data.
    StreamId stream_id;

    //! Number of state elements
    CELER_FUNCTION size_type size() const { return particles.size(); }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && materials && particles && physics && rng && sim
               && init && stream_id;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CoreStateData& operator=(CoreStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry = other.geometry;
        materials = other.materials;
        particles = other.particles;
        physics = other.physics;
        rng = other.rng;
        sim = other.sim;
        init = other.init;
        track_slots = other.track_slots;
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
}  // namespace celeritas
