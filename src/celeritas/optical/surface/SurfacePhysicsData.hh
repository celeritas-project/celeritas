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

using SurfaceId = OpaqueId<struct OpticalSurface_>;

using ValueGrid = NonuniformGridRecord;
using ValudGridId = OpaqueId<ValueGrid>;

//---------------------------------------------------------------------------//
/*!
 * Store optical physics data for a given surface layer.
 */
struct SurfaceLayerRecord
{
    // TODO: Per model layer IDs

    ActionId roughness_model{};
    ActionId reflectivity_model{};
    ActionId interaction_model{};

    //! Whether data is assigned and valid
    inline CELER_FUNCTION operator bool() const { return false; }
};

/*!
 * Store interleaved layers and optical materials that constitute a surface.
 *
 * If there are N surface layers then there should be N+1 materials. The ith
 * layer separates the ith and (i+1)th materials.
 */
struct SurfaceRecord
{
    ItemRange<SurfaceLayerRecordId> layers;
    ItemRange<OpticalMaterialId> optical_materials;
};

//---------------------------------------------------------------------------//
/*!
 * Scalar quantities used by optical surface physics.
 */
struct SurfacePhysicsParamsScalars
{
    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const {}
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

    //! Non-templated data
    SurfacePhysicsParamsScalars scalars;

    //! Surface data
    SurfaceItems<SurfaceRecord> surfaces;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(scalars);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfacePhysicsParamsData<W, M>&
    operator=(SurfacePhysicsParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        scalars = other.scalars;
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
    StateItems<SurfaceLayerId> current_layer;
    StateItems<Real3> surface_normal;
    StateItems<Real3> layer_normal;

    //// Temporary State Data ////

    StateItems<Real3> facet_normal;
    StateItems<real_type> reflectivity;

    //// Methods ////

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !surface_normal.empty();
    }

    //! State size
    CELER_FUNCTION size_type size() const { return surface_normal.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfacePhysicsStateData<W, M>&
    operator=(SurfacePhysicsStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        surface_normal = other.surface_normal;
        facet_normal = other.facet_normal;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize the state in host code.
 */
template<MemSpace M>
inline void resize(PhysicsStateData<Ownership::value, M>* state, size_type size)
{
    CELER_EXPECT(state);
    CELER_EXPECT(size > 0);

    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
