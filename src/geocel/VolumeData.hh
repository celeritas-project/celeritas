//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "geocel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Unique instance ID of the "world" volume (root of the volume graph)
inline constexpr VolumeUniqueInstanceId world_unique_instance{0};

//---------------------------------------------------------------------------//
/*!
 * Per-volume data: material assignment and parent/child instance edges.
 *
 * Parents and children are stored as ranges into the \c vi_storage backend of
 * \c VolumeParamsData.
 */
struct VolumeRecord
{
    //! Material assignment for this volume
    GeoMatId material;
    //! Incoming edges: volume instances that instantiate this volume
    ItemRange<VolumeInstanceId> parents;
    //! Outgoing edges: child volume instances placed inside this volume
    ItemRange<VolumeInstanceId> children;
};

//---------------------------------------------------------------------------//
/*!
 * Scalar values for the volume hierarchy graph.
 */
struct VolumeParamsScalars
{
    //! Root volume of the geometry graph
    VolumeId world;
    //! Enclosing instance of the world volume (null if world is a true root)
    VolumeInstanceId world_instance;
    //! Number of logical volumes (nodes)
    VolumeId::size_type num_volumes{0};
    //! Number of volume instances (edges)
    VolumeInstanceId::size_type num_volume_instances{0};
    //! Total number of unique root-to-node paths (= num_desc of world volume)
    VolumeUniqueInstanceId::size_type num_unique_instances{0};
    //! Depth of the volume graph (1 for a world with no children)
    VolumeLevelId::size_type num_volume_levels{0};

    //! True if scalars are in a consistent populated state
    explicit CELER_FUNCTION operator bool() const
    {
        return num_volume_levels > 0 && num_unique_instances > 0
               && static_cast<bool>(world);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent data for the volume hierarchy graph.
 *
 * This stores the volume DAG (directed acyclic graph) in Collection-based
 * storage for host and device compatibility. Volumes are nodes and volume
 * instances are edges.
 *
 * See \c VolumeParams for construction details and label-based lookup.
 */
template<Ownership W, MemSpace M>
struct VolumeParamsData
{
    //// TYPES ////

    template<class T>
    using VolumeItems = Collection<T, W, M, VolumeId>;
    template<class T>
    using VolInstItems = Collection<T, W, M, VolumeInstanceId>;
    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    //! Per-volume records (material, parent/child ranges into vi_storage)
    VolumeItems<VolumeRecord> volumes;
    //! Logical volume for each volume instance (vi -> v)
    VolInstItems<VolumeId> volume_ids;
    //! Flat backing storage for per-volume parent and child instance lists
    Items<VolumeInstanceId> vi_storage;
    //! Pre-computed per-sibling offset (see \c VolumePathAccumulator)
    VolInstItems<VolumeUniqueInstanceId::size_type> unique_instance_offsets;

    //! Scalar values (world ID, world instance, depths, etc.)
    VolumeParamsScalars scalars;

    //// METHODS ////

    //! True if data is consistent (empty and fully-populated states are valid)
    explicit CELER_FUNCTION operator bool() const
    {
        if (volumes.empty())
        {
            // Valid empty state (e.g., for testing or ORANGE debugging)
            return volume_ids.empty() && vi_storage.empty()
                   && unique_instance_offsets.empty() && !scalars;
        }
        return static_cast<bool>(scalars)
               && unique_instance_offsets.size() == volume_ids.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    VolumeParamsData& operator=(VolumeParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        volumes = other.volumes;
        volume_ids = other.volume_ids;
        vi_storage = other.vi_storage;
        unique_instance_offsets = other.unique_instance_offsets;
        scalars = other.scalars;
        CELER_ENSURE(*this);
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
