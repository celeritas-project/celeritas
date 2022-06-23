//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/VecgeomData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Types.hh"

#include "detail/VecgeomNavCollection.hh"
#include "detail/VecgeomTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

//! Identifier for a geometry volume
using VolumeId = OpaqueId<struct Volume>;

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

    const PlacedVolumeT* world_volume = nullptr;
    int                  max_depth    = 0;

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
        max_depth    = other.max_depth;
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
    Items<Real3>     pos;
    Items<Real3>     dir;
    Items<real_type> next_step;

    // Wrapper for NavStatePool, vector, or void*
    detail::VecgeomNavCollection<W, M> vgstate;
    detail::VecgeomNavCollection<W, M> vgnext;

    //// METHODS ////

    //! True if sizes are consistent and states are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return this->size() > 0 && dir.size() == this->size()
               && next_step.size() == this->size() && vgstate && vgnext;
    }

    //! State size
    CELER_FUNCTION ThreadId::size_type size() const { return pos.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    VecgeomStateData& operator=(VecgeomStateData<W2, M2>& other)
    {
        static_assert(M2 == M && W2 == Ownership::value
                          && W == Ownership::reference,
                      "Only supported assignment is from value to reference");
        CELER_EXPECT(other);
        pos       = other.pos;
        dir       = other.dir;
        next_step = other.next_step;
        vgstate   = other.vgstate;
        vgnext    = other.vgnext;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize geometry states.
 */
template<MemSpace M>
void resize(
    VecgeomStateData<Ownership::value, M>*                               data,
    const VecgeomParamsData<Ownership::const_reference, MemSpace::host>& params,
    size_type                                                            size)
{
    CELER_EXPECT(data);
    CELER_EXPECT(size > 0);
    CELER_EXPECT(params.max_depth > 0);

    resize(&data->pos, size);
    resize(&data->dir, size);
    resize(&data->next_step, size);
    data->vgstate.resize(params.max_depth, size);
    data->vgnext.resize(params.max_depth, size);

    CELER_ENSURE(data);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
