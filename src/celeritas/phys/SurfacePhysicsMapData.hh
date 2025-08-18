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
 * \todo generalize to other ID types/subtypes and use for volumetric physics
 * to eliminate some of the complexity?
 */
template<Ownership W, MemSpace M>
struct SurfacePhysicsMapData
{
    //// TYPES ////

    using SurfaceLayer = SurfaceModel::SurfaceLayer;
    using ModelSurfaceId = SurfaceModel::ModelSurfaceId;
    template<class T>
    using SurfaceItems = Collection<T, W, M, SurfaceId>;

    //// DATA ////

    SurfaceItems<ActionId> action_ids;
    SurfaceItems<ModelSurfaceId> model_surface_ids;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !action_ids.empty()
               && action_ids.size() == model_surface_ids.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfacePhysicsMapData& operator=(SurfacePhysicsMapData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        action_ids = other.action_ids;
        model_surface_ids = other.model_surface_ids;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
