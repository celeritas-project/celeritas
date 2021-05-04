//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Collection.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "detail/VGNavCollection.hh"
#include "detail/VGTraits.hh"

#if !CELER_SHIELD_DEVICE
#    include "base/CollectionBuilder.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Pointers to persistent data used by VecGeom implementation.
 *
 * If the GeoParamsPointers is constructed by \c VGHost::host_pointers, it
 * points to a \c vecgeom::cxx::VPlacedVolume . If built by \c
 * VGDevice::device_pointers, it points to a \c vecgeom::cuda::VPlacedVolume .
 *
 * Note that because of VecGeom default namespaces triggered by the presence of
 * the \c __NVCC__ macro, this data structure actually has different types
 * <em>depending on what compiler is active</em>. Since the \c GeoTrackView
 * implementation is designed to work with both CPU and GPU (depending on
 * \c __CUDA_ARCH__ and whether the code is on device, rather than the \c
 * __NVCC__ compiler) we can't simply declare this pointer to be in the \c cuda
 * or \c cxx explicit namespaces.
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
 * View to a vector of VecGeom state information.
 *
 * This "view" is expected to be an argument to a geometry-related kernel
 * launch. It contains pointers to host-managed data.
 *
 * The \c vgstate and \c vgnext arguments must be the result of
 * vecgeom::NavStateContainer::GetGPUPointer; and they are only meaningful with
 * the corresponding \c vgmaxdepth, the result of \c GeoManager::getMaxDepth .
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

#if !CELER_SHIELD_DEVICE
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
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
