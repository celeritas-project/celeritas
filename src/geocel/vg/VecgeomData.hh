//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/Types.hh"

#include "detail/VecgeomNavCollection.hh"
#include "detail/VecgeomTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//

//! VecGeom::VPlacedVolume::id is unsigned int
using VecgeomPlacedVolumeId
    = OpaqueId<struct VecgeomPlacedVolume_, unsigned int>;

//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
struct VecgeomScalars
{
    template<MemSpace M>
    using PlacedVolumeT = typename detail::VecgeomTraits<M>::PlacedVolume;

    PlacedVolumeT<MemSpace::host> const* host_world{nullptr};
    PlacedVolumeT<MemSpace::device> const* device_world{nullptr};
    LevelId::size_type max_depth = 0;

    template<MemSpace M>
    CELER_FUNCTION PlacedVolumeT<M> const* world() const
    {
        if constexpr (M == MemSpace::host)
        {
            return host_world;
        }
        else if constexpr (M == MemSpace::device)
        {
            return device_world;
        }
#if CELER_CUDACC_BUGGY_IF_CONSTEXPR
        CELER_ASSERT_UNREACHABLE();
#endif
    }

    //! Whether the scalars are valid (device may be null)
    explicit CELER_FUNCTION operator bool() const
    {
        return host_world != nullptr && max_depth > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent data used by VecGeom implementation.
 *
 * The volumes and volume instance mappings are set when constructing from an
 * external model, using VolumeParams to map to Geant4 geometry. For models
 * loaded through VGDML, the mapping is currently one-to-one for implementation
 * and placed volumes.
 */
template<Ownership W, MemSpace M>
struct VecgeomParamsData
{
    using ImplVolInstanceId = VecgeomPlacedVolumeId;

    //! Values that don't require host/device copying
    VecgeomScalars scalars;

    //! Map logical volume ID to canonical
    Collection<VolumeId, W, M, ImplVolumeId> volumes;
    //! Map placed volume ID to canonical
    Collection<VolumeInstanceId, W, M, ImplVolInstanceId> volume_instances;

    //! Whether the data is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        // Volumes/instances can be absent if built from GDML
        return scalars && !volumes.empty() && !volume_instances.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    VecgeomParamsData& operator=(VecgeomParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        scalars = other.scalars;
        volumes = other.volumes;
        volume_instances = other.volume_instances;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Interface for VecGeom state information.
 */
template<Ownership W, MemSpace M>
struct VecgeomStateData
{
    //// TYPES ////

    template<class T>
    using Items = celeritas::StateCollection<T, W, M>;

    //// DATA ////

    // Collections
    Items<Real3> pos;
    Items<Real3> dir;
#if CELERITAS_VECGEOM_SURFACE
    Items<long> next_surface;
#endif

    // Wrapper for NavStatePool, vector, or void*
    detail::VecgeomNavCollection<W, M> vgstate;
    detail::VecgeomNavCollection<W, M> vgnext;

    //// METHODS ////

    //! True if sizes are consistent and states are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return this->size() > 0 && dir.size() == this->size()
#if CELERITAS_VECGEOM_SURFACE
               && next_surface.size() == this->size()
#endif
               && vgstate && vgnext;
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const { return pos.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    VecgeomStateData& operator=(VecgeomStateData<W2, M2>& other)
    {
        static_assert(M2 == M && W == Ownership::reference,
                      "Only supported assignment is from the same memspace to "
                      "a reference");
        CELER_EXPECT(other);
        pos = other.pos;
        dir = other.dir;
#if CELERITAS_VECGEOM_SURFACE
        next_surface = other.next_surface;
#endif
        vgstate = other.vgstate;
        vgnext = other.vgnext;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize geometry states.
 */
template<MemSpace M>
void resize(VecgeomStateData<Ownership::value, M>* data,
            HostCRef<VecgeomParamsData> const& params,
            size_type size)
{
    CELER_EXPECT(data);
    CELER_EXPECT(size > 0);
    CELER_EXPECT(params);

    resize(&data->pos, size);
    resize(&data->dir, size);
#if CELERITAS_VECGEOM_SURFACE
    resize(&data->next_surface, size);
#endif
    data->vgstate.resize(params.scalars.max_depth, size);
    data->vgnext.resize(params.scalars.max_depth, size);

    CELER_ENSURE(data);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
