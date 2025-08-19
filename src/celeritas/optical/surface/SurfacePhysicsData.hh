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

using GeometricSurfaceId = OpaqueId<struct GeometricSurface_>;
using PhysicsSurfaceId = OpaqueId<struct PhysicsSurface_>;
using SubsurfaceMaterialId = OpaqueId<struct SubsurfaceMaterial_>;
using SubsurfaceInterfaceId = OpaqueId<struct SubsurfaceInterface_>;
using SubsurfaceMaterialRecord = OptMatId;  // OpaqueId<struct
                                            // SubsurfaceMaterialRecord>;
using SubsurfaceMaterialRecordId = OpaqueId<SubsurfaceMaterialRecord>;
using SubsurfaceInterfaceRecord
    = PhysicsSurfaceId;  // OpaqueId<struct SubsurfaceInterfaceRecord>;
using SubsurfaceInterfaceRecordId = OpaqueId<SubsurfaceInterfaceRecord>;
using SurfaceTrackPosition = OpaqueId<struct SurfaceTrackPosition_>;

struct SurfaceRecord
{
    ItemMap<SubsurfaceMaterialId, SubsurfaceMaterialRecordId> subsurface_materials;
    ItemMap<SubsurfaceInterfaceId, SubsurfaceInterfaceRecordId>
        subsurface_interfaces;

    //! Whether data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !subsurface_interfaces.empty()
               && subsurface_materials.size()
                      == subsurface_interfaces.size() + 1;
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
    using ModelMap = SurfacePhysicsMapData<W, M>;

    template<class T>
    using GeoSurfaceItems = Collection<T, W, M, GeometricSurfaceId>;

    template<class T>
    using SurfaceStepArray = EnumArray<SurfacePhysicsStep, T>;

    template<class T>
    using Items = Collection<T, W, M>;
    //!@}

    GeoSurfaceItems<SurfaceRecord> surfaces;
    SurfaceStepArray<ModelMap> model_maps;
    Items<SubsurfaceMaterialRecord> subsurface_materials;
    Items<SubsurfaceInterfaceRecord> subsurface_interfaces;

    //! Whether data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !surfaces.empty() && !subsurface_materials.empty()
               && !subsurface_interfaces.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfacePhysicsParamsData<W, M>&
    operator=(SurfacePhysicsParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        surfaces = other.surfaces;
        subsurface_materials = other.subsurface_materials;
        subsurface_interfaces = other.subsurface_interfaces;
        for (auto step : range(SurfacePhysicsStep::size_))
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

    StateItems<GeometricSurfaceId> surface;
    StateItems<SubsurfaceDirection> surface_orientation;
    StateItems<SurfaceTrackPosition> surface_position;

    //! Whether data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !surface.empty() && surface.size() == surface_orientation.size()
               && surface.size() == surface_position.size();
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
        surface_position = other.surface_position;
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
    resize(&state->surface_position, size);

    CELER_ENSURE(*state);
    CELER_ENSURE(state->size() == size);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
