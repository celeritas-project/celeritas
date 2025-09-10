//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/optical/Types.hh"
#include "celeritas/phys/SurfacePhysicsMapData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using SurfaceTrackPosition = OpaqueId<struct SurfaceTrackPosition_>;
using SubsurfaceMaterialId = OpaqueId<struct SubsurfaceMaterial_>;
using SubsurfaceInterfaceId = OpaqueId<struct SubsurfaceInterface_>;

//---------------------------------------------------------------------------//
/*!
 * Storage for physics data of a geometric surface.
 *
 * The \c subsurface_materials indexes into the \c
 * SurfacePhysicsParamsData::subsurface_materials and represents a list of
 * interstitial optical materials that make up a geometric surface. The \c
 * subsurface_interfaces represents the physics surfaces that separate the
 * optical materials that make up a geometric surface.
 *
 * By convention, \c subsurface_interfaces[0] separates the pre-volume and the
 * first subsurface material, while the last interface separates the last
 * subsurface material and the post-volume.
 */
struct SurfaceRecord
{
    ItemMap<SubsurfaceMaterialId, OpaqueId<OptMatId>> subsurface_materials;
    ItemMap<SubsurfaceInterfaceId, PhysSurfaceId> subsurface_interfaces;

    //! Whether data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return subsurface_materials.size() + 1 == subsurface_interfaces.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Scalar quantities used by optical surface physics.
 */
struct SurfacePhysicsParamsScalars
{
    //! ID of the default surface
    SurfaceId default_surface{};

    //! Action ID of the init-boundary action
    ActionId init_boundary_action{};

    //! Action ID of the surface stepping loop action
    ActionId surface_stepping_action{};

    //! Action ID of the post-boundary action
    ActionId post_boundary_action{};

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return default_surface && post_boundary_action
               && surface_stepping_action < post_boundary_action
               && init_boundary_action < surface_stepping_action;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent shared optical surface data
 */
template<Ownership W, MemSpace M>
struct SurfacePhysicsParamsData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using GeoSurfaceItems = Collection<T, W, M, SurfaceId>;

    using SurfaceStepModelMaps
        = EnumArray<SurfacePhysicsOrder, SurfacePhysicsMapData<W, M>>;

    template<class T>
    using Items = Collection<T, W, M>;
    //!@}

    //! Non-templated data
    SurfacePhysicsParamsScalars scalars;

    //! Optical surface model data

    GeoSurfaceItems<SurfaceRecord> surfaces;
    SurfaceStepModelMaps model_maps;
    Items<OptMatId> subsurface_materials;

    //! Whether data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return scalars && !surfaces.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfacePhysicsParamsData<W, M>&
    operator=(SurfacePhysicsParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        scalars = other.scalars;
        surfaces = other.surfaces;
        subsurface_materials = other.subsurface_materials;
        for (auto step : range(SurfacePhysicsOrder::size_))
        {
            model_maps[step] = other.model_maps[step];
        }
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Dynamic optical surface physics state data.
 */
template<Ownership W, MemSpace M>
struct SurfacePhysicsStateData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using StateItems = StateCollection<T, W, M>;
    //!@}

    StateItems<SurfaceId> surface;
    StateItems<SubsurfaceDirection> surface_orientation;
    StateItems<Real3> global_normal;
    StateItems<OptMatId> pre_volume_material;
    StateItems<OptMatId> post_volume_material;

    StateItems<SurfaceTrackPosition> surface_position;
    StateItems<Real3> facet_normal;

    //! Whether data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !surface.empty() && surface.size() == surface_orientation.size()
               && surface.size() == surface_position.size()
               && surface.size() == pre_volume_material.size()
               && surface.size() == post_volume_material.size()
               && surface.size() == global_normal.size()
               && surface.size() == facet_normal.size();
    }

    //! State size
    CELER_FUNCTION size_type size() const { return surface.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfacePhysicsStateData<W, M>&
    operator=(SurfacePhysicsStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        surface = other.surface;
        surface_orientation = other.surface_orientation;
        global_normal = other.global_normal;
        surface_position = other.surface_position;
        pre_volume_material = other.pre_volume_material;
        post_volume_material = other.post_volume_material;
        facet_normal = other.facet_normal;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize the state in host code.
 */
template<MemSpace M>
inline void
resize(SurfacePhysicsStateData<Ownership::value, M>* state, size_type size)
{
    CELER_EXPECT(state);
    CELER_EXPECT(size > 0);

    resize(&state->surface, size);
    resize(&state->surface_orientation, size);
    resize(&state->global_normal, size);
    resize(&state->surface_position, size);
    resize(&state->pre_volume_material, size);
    resize(&state->post_volume_material, size);
    resize(&state->facet_normal, size);

    CELER_ENSURE(*state);
    CELER_ENSURE(state->size() == size);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
