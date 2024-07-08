//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalTrackData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/random/RngData.hh"
#include "celeritas/track/SimData.hh"
#include "celeritas/track/TrackInitData.hh"

#include "Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
// XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX
// IMPLEMENT ME!

template<Ownership W, MemSpace M>
struct OpticalPhysicsParamsData
{
    explicit CELER_FUNCTION operator bool() const { return false; }
};
template<Ownership W, MemSpace M>
struct OpticalPhysicsStateData
{
};
template<MemSpace M>
void resize(OpticalPhysicsStateData<Ownership::value, M>*,
            HostCRef<OpticalPhysicsParamsData> const&,
            size_type)
{
    CELER_NOT_IMPLEMENTED("optical physics state");
}

// XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX
//---------------------------------------------------------------------------//
/*!
 * Memspace-independent core variables.
 */
struct OpticalScalars
{
    ActionId boundary_action;

    StreamId::size_type max_streams{0};
    OpticalMaterialId::size_type num_materials{0};

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return boundary_action && max_streams > 0 && num_materials > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Immutable problem data.
 */
template<Ownership W, MemSpace M>
struct OpticalParamsData
{
    template<class T>
    using VolumeItems = celeritas::Collection<T, W, M, VolumeId>;

    GeoParamsData<W, M> geometry;
    VolumeItems<OpticalMaterialId> materials;
    OpticalPhysicsParamsData<W, M> physics;
    RngParamsData<W, M> rng;
    TrackInitParamsData<W, M> init;  // TODO: don't need max events

    OpticalScalars scalars;

    //! True if all params are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && !materials.empty() && physics && rng && init
               && scalars;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OpticalParamsData& operator=(OpticalParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        geometry = other.geometry;
        materials = other.materials;
        physics = other.physics;
        rng = other.rng;
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
struct OpticalStateData
{
    template<class T>
    using Items = StateCollection<T, W, M>;

    GeoStateData<W, M> geometry;
    Items<OpticalMaterialId> materials;
    OpticalPhysicsStateData<W, M> physics;
    RngStateData<W, M> rng;
    SimStateData<W, M> sim;  // TODO: has a few things we don't need
    TrackInitStateData<W, M> init;  // Still need to track vacancies

    //! Unique identifier for "thread-local" data.
    StreamId stream_id;

    //! Number of state elements
    CELER_FUNCTION size_type size() const { return geometry.size(); }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && materials && physics && rng && sim && init
               && stream_id;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OpticalStateData& operator=(OpticalStateData<W2, M2>& other)
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
void resize(OpticalStateData<Ownership::value, M>* state,
            HostCRef<OpticalParamsData> const& params,
            StreamId stream_id,
            size_type size);

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
