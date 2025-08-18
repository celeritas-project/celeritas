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
 * Device-compatible map between surface+layer IDs and actions/indices.
 *
 * One or more instances of these should be stored as member data inside a
 * downstream ParamsData class. For instance, optical surface physics will have
 * one map for roughness, one for reflectivity, and one for interaction.
 *
 * \todo support for layers
 */
template<Ownership W, MemSpace M>
struct SurfacePhysicsMapData
{
    //// TYPES ////

    using SurfaceLayer = SurfaceModel::SurfaceLayer;
    using SurfaceModelId = SurfaceModel::SurfaceModelId;
    using InternalSurfaceId = SurfaceModel::InternalSurfaceId;
    template<class T>
    using SurfaceItems = Collection<T, W, M, SurfaceId>;

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
