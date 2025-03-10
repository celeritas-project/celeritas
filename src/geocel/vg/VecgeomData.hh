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
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Persistent data used by VecGeom implementation.
 */
template<Ownership W, MemSpace M>
struct VecgeomParamsData
{
    using PlacedVolumeT = typename detail::VecgeomTraits<M>::PlacedVolume;

    PlacedVolumeT const* world_volume = nullptr;
    LevelId::size_type max_depth = 0;

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return world_volume != nullptr && max_depth > 0;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    VecgeomParamsData& operator=(VecgeomParamsData<W2, M2>& other)
    {
        static_assert(M2 == M && W2 == Ownership::value
                          && W == Ownership::reference,
                      "Only supported assignment is from value to reference");
        CELER_EXPECT(other);
        world_volume = other.world_volume;
        max_depth = other.max_depth;
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
    CELER_EXPECT(params.max_depth > 0);

    resize(&data->pos, size);
    resize(&data->dir, size);
#if CELERITAS_VECGEOM_SURFACE
    resize(&data->next_surface, size);
#endif
    data->vgstate.resize(params.max_depth, size);
    data->vgnext.resize(params.max_depth, size);

    CELER_ENSURE(data);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
