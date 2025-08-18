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
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    SurfacePhysicsData ...;
   \endcode
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
    explicit CELER_FUNCTION operator bool() const { return false; }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfacePhysicsParamsData<W, M>&
    operator=(SurfacePhysicsParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        return *this;
    }
};

template<Ownership W, MemSpace M>
struct SurfacePhysicsStateData
{
    //!@{
    //! \name Type aliases
    //!@}

    //! Whether data is assigned
    explicit CELER_FUNCTION operator bool() const { return false; }

    //! State size
    CELER_FUNCTION size_type size() const { return 0; }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfacePhysicsStateData<W, M>&
    operator=(SurfacePhysicsStateData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        return *this;
    }
};

template<MemSpace M>
inline void resize(SurfacePhysicsStateData<Ownership::value, M>* /* state */,
                   size_type /* size */)
{
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
