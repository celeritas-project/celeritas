//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Collection.hh"
#include "base/CollectionBuilder.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "detail/VGNavCollection.hh"
#include "detail/VGTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Persistent data used by VecGeom implementation.
 */
template<Ownership W, MemSpace M>
struct GeoParamsData
{
    using PlacedVolumeT = typename detail::VGTraits<M>::PlacedVolume;

    const PlacedVolumeT* world_volume = nullptr;
    int                  max_depth    = 0;

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return world_volume != nullptr && max_depth > 0;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    GeoParamsData& operator=(GeoParamsData<W2, M2>& other)
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
 * Data required to initialize a geometry state.
 */
struct GeoTrackInitializer
{
    Real3 pos;
    Real3 dir;
};

//---------------------------------------------------------------------------//
/*!
 * Interface for VecGeom state information.
 */
template<Ownership W, MemSpace M>
struct GeoStateData
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
    detail::VGNavCollection<W, M> vgstate;
    detail::VGNavCollection<W, M> vgnext;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return this->size() > 0 && dir.size() == this->size()
               && next_step.size() == this->size() && vgstate && vgnext;
    }

    //! State size
    CELER_FUNCTION ThreadId::size_type size() const { return pos.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    GeoStateData& operator=(GeoStateData<W2, M2>& other)
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
 * Resize particle states in host code.
 */
template<MemSpace M>
void resize(
    GeoStateData<Ownership::value, M>*                               data,
    const GeoParamsData<Ownership::const_reference, MemSpace::host>& params,
    size_type                                                        size)
{
    CELER_EXPECT(data);
    CELER_EXPECT(size > 0);
    CELER_EXPECT(params.max_depth > 0);

    make_builder(&data->pos).resize(size);
    make_builder(&data->dir).resize(size);
    make_builder(&data->next_step).resize(size);
    data->vgstate.resize(params.max_depth, size);
    data->vgnext.resize(params.max_depth, size);

    CELER_ENSURE(data);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
