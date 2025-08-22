//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/SurfacePhysicsMapData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"

#include "SurfaceModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Device-compatible map between physics surface IDs and models/indices.
 *
 * One or more instances of these should be stored as member data inside a
 * downstream ParamsData class. For instance, optical surface physics will have
 * one map for roughness, one for reflectivity, and one for interaction.
 *
 * If a \c SurfaceModel with ID 10 returns a list of surfaces `{3, 1, 5}` and
 * another with ID 11 returns `{{}, 0, 4}`, then this class will have a
 * key-value mapping stored as two arrays:
 * \code
 * surface_models = {11, 10, <null>, 10, 11, 10, 11};
 * internal_surface_ids = {1, 1, <null>, 0, 2, 2, 0};
 * \endcode
 *
 * Note that the "default" surface (the empty item returned by the second
 * surface model) becomes an additional pseudo-surface at the end of the array.
 * <b>The surface physics will always have one more surface entry than the
 * actual geometry.</b> \todo This will change when we map geometric surfaces
 * to vectors of physics surface interfaces.
 *
 * With this setup, \c Collection data can be accessed locally by indexing on
 * \c ModelSurfaceId .
 */
template<Ownership W, MemSpace M>
struct SurfacePhysicsMapData
{
    //// TYPES ////

    using PhysSurfaceId = SurfaceModel::PhysSurfaceId;
    using SurfaceModelId = SurfaceModel::SurfaceModelId;
    using InternalSurfaceId = SurfaceModel::InternalSurfaceId;
    template<class T>
    using SurfaceItems = Collection<T, W, M, PhysSurfaceId>;

    //// DATA ////

    SurfaceItems<SurfaceModelId> surface_models;
    SurfaceItems<InternalSurfaceId> internal_surface_ids;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !surface_models.empty()
               && surface_models.size() == internal_surface_ids.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfacePhysicsMapData& operator=(SurfacePhysicsMapData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        surface_models = other.surface_models;
        internal_surface_ids = other.internal_surface_ids;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
