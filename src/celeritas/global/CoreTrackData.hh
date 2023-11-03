//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreTrackData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoMaterialData.hh"
#include "celeritas/mat/MaterialData.hh"
#include "celeritas/phys/CutoffData.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/PhysicsData.hh"
#include "celeritas/random/RngData.hh"
#include "celeritas/track/SimData.hh"
#include "celeritas/track/TrackInitData.hh"
#include "celeritas/user/DiagnosticData.hh"

#include "CoreTrackDataFwd.hh"

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
    ActionId abandon_looping_action;

    // TODO: this is a hack until we improve the along-step interface
    ActionId along_step_user_action;
    ActionId along_step_neutral_action;

    StreamId::size_type max_streams{0};

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return boundary_action && propagation_limit_action
               && abandon_looping_action && along_step_user_action
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
    GeoMaterialParamsData<W, M> geo_mats;
    MaterialParamsData<W, M> materials;
    ParticleParamsData<W, M> particles;
    CutoffParamsData<W, M> cutoffs;
    PhysicsParamsData<W, M> physics;
    RngParamsData<W, M> rng;
    SimParamsData<W, M> sim;
    TrackInitParamsData<W, M> init;
    DiagnosticParamsData<W, M> diagnostic;

    CoreScalars scalars;

    //! True if all params are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && geo_mats && materials && particles && cutoffs
               && physics && sim && init && diagnostic && scalars;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CoreParamsData& operator=(CoreParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        geometry = other.geometry;
        geo_mats = other.geo_mats;
        materials = other.materials;
        particles = other.particles;
        cutoffs = other.cutoffs;
        physics = other.physics;
        rng = other.rng;
        sim = other.sim;
        init = other.init;
        diagnostic = other.diagnostic;
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
    using Items = StateCollection<T, W, M>;

    template<class T>
    using ThreadItems = Collection<T, W, M, ThreadId>;

    GeoStateData<W, M> geometry;
    MaterialStateData<W, M> materials;
    ParticleStateData<W, M> particles;
    PhysicsStateData<W, M> physics;
    RngStateData<W, M> rng;
    SimStateData<W, M> sim;
    TrackInitStateData<W, M> init;
    DiagnosticStateData<W, M> diagnostic;
    ThreadItems<TrackSlotId::size_type> track_slots;

    //! Unique identifier for "thread-local" data.
    StreamId stream_id;

    //! Number of state elements
    CELER_FUNCTION size_type size() const { return particles.size(); }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && materials && particles && physics && rng && sim
               && init && diagnostic && stream_id;
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
        diagnostic = other.diagnostic;
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
