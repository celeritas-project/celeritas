//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using ValueGrid = NonuniformGridRecord;
using ValudGridId = OpaqueId<ValueGrid>;

//---------------------------------------------------------------------------//
/*!
 * Store optical physics data for a given surface.
 */
struct SurfaceRecord
{
    ActionId roughness_model{};
    ActionId reflectivity_model{};
    ActionId interaction_model{};

    //! Whether data is assigned and valid
    inline CELER_FUNCTION operator bool() const
    {
        return roughness_model && reflectivity_model && interaction_model;
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

    //! Surface data
    SurfaceItems<SurfaceRecord> surfaces;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const { return true; }

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
    StateItems<Real3> surface_normal;

    //// Temporary State Data ////

    StateItems<Real3> facet_normal;
    StateItems<real_type> reflectivity;

    //// Methods ////

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !surface.empty() && !surface_normal.empty()
               && !facet_normal.empty() && !reflectivity.empty();
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
        surface_normal = other.surface_normal;
        facet_normal = other.facet_normal;
        reflectivity = other.reflectivity;
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
    resize(&state->surface_normal, size);
    resize(&state->facet_normal, size);
    resize(&state->reflectivity, size);

    CELER_ENSURE(*state);
    CELER_ENSURE(state->size() == size);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
