//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/TrackData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/random/RngData.hh"
#include "celeritas/track/SimData.hh"

#include "MaterialData.hh"
#include "TrackInitData.hh"
#include "Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
// XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX
// IMPLEMENT ME!

template<Ownership W, MemSpace M>
struct PhysicsParamsData
{
    explicit CELER_FUNCTION operator bool() const { return true; }
};
template<Ownership W, MemSpace M>
struct PhysicsStateData
{
    explicit CELER_FUNCTION operator bool() const { return true; }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    PhysicsStateData& operator=(PhysicsStateData<W2, M2>&)
    {
        return *this;
    }
};

template<MemSpace M>
inline void resize(PhysicsStateData<Ownership::value, M>*,
                   HostCRef<PhysicsParamsData> const&,
                   size_type)
{
}

// XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX
//---------------------------------------------------------------------------//
/*!
 * Memspace-independent core variables.
 */
struct CoreScalars
{
    // TODO: maybe replace with a surface crossing manager to handle boundary
    // conditions (see CoreParams.cc)
    ActionId boundary_action;

    StreamId::size_type max_streams{0};

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return boundary_action && max_streams > 0;
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
    MaterialParamsData<W, M> material;
    PhysicsParamsData<W, M> physics;
    RngParamsData<W, M> rng;
    SimParamsData<W, M> sim;
    TrackInitParamsData<W, M> init;

    CoreScalars scalars;

    //! True if all params are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && material && physics && rng && sim && init && scalars;
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
        init = other.init;
        scalars = other.scalars;
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
    Items<OpticalMaterialId> materials;
    PhysicsStateData<W, M> physics;
    RngStateData<W, M> rng;
    SimStateData<W, M> sim;  // TODO: has a few things we don't need
    TrackInitStateData<W, M> init;

    //! Unique identifier for "thread-local" data.
    StreamId stream_id;

    //! Number of state elements
    CELER_FUNCTION size_type size() const { return geometry.size(); }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && !materials.empty() && physics && rng && sim && init
               && stream_id;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CoreStateData& operator=(CoreStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry = other.geometry;
        materials = other.materials;
        physics = other.physics;
        rng = other.rng;
        sim = other.sim;
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
