//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//

using SubsurfaceMaterialRecordId = OpaqueId<struct SubsurfaceMaterialRecord>;
using SubsurfaceInterfaceRecordId = OpaqueId<struct SubsurfaceInterfaceRecord>;

struct SubsurfaceMaterialRecord
{
    // TODO: Add subsurface material property data
};

struct SubsurfaceInterfaceRecord
{
    // TODO: Add subsurface interface property data
};

//---------------------------------------------------------------------------//
/*!
 * Storage for surface physics data.
 *
 * A surface between volumes A and B is an ordered list of N \c
 * SubsurfaceInterface that separate N+1 \c SubsurfaceMaterial . The data is
 * organized as:
 *  - \c SubsurfaceMaterial 0 corresponds to material properties of A
 *  - \c SubsurfaceMaterial N corresponds to material properties of B
 *  - \c SubsurfaceInterface i corresponds to the interface between \c
 * SubsurfaceMaterial i and i+1.
 *  - There is always at least one \c SubsurfaceInterface i.e. N >= 1
 */
struct SurfacePhysicsRecord
{
    ItemMap<SubsurfaceMaterialId, SubsurfaceMaterialRecordId> subsurface_materials;
    ItemMap<SubsurfaceInterfaceId, SubsurfaceInterfaceRecordId>
        subsurface_interfaces;

    explicit CELER_FUNCTION operator bool() const
    {
        return !subsurface_interfaces.empty()
               && subsurface_interfaces.size() + 1
                      == subsurface_materials.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent shared optical surface physics data.
 */
template<Ownership W, MemSpace M>
struct SurfacePhysicsParamsData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, W, M>;

    template<class T>
    using SurfaceItems = Collection<T, W, M, SurfaceId>;
    //!@}

    SurfaceItems<SurfacePhysicsRecord> surfaces;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const { return !surfaces.empty(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfacePhysicsParamsData<W, M>&
    operator=(SurfacePhysicsParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        surfaces = other.surfaces;
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
    using Items = Collection<T, W, M>;
    template<class T>
    using StateItems = StateCollection<T, W, M>;
    //!@}

    //// Persistent State Data ////

    StateItems<SurfaceId> surface;
    StateItems<SubsurfaceDirection> surface_orientation;
    StateItems<SubsurfaceMaterialId> subsurface_material;

    //// Methods ////

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !surface.empty() && !subsurface_material.empty()
               && !surface_orientation.empty();
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
        subsurface_material = other.subsurface_material;
        surface_orientation = other.surface_orientation;
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
    resize(&state->subsurface_material, size);
    resize(&state->surface_orientation, size);

    CELER_ENSURE(*state);
    CELER_ENSURE(state->size() == size);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
