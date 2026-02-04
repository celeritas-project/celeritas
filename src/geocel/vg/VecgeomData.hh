//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/Version.h>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/Types.hh"

#include "VecgeomTypes.hh"

#if CELERITAS_VECGEOM_VERSION >= 0x020000 && defined(VECGEOM_ENABLE_CUDA)
// IWYU errors from navindex/tuple
#    include <VecGeom/base/Cuda.h>
#    include <VecGeom/base/Global.h>
#    include <VecGeom/management/CudaManager.h>
#endif

#if CELER_VGNAV == CELER_VGNAV_PATH
#    include "detail/VecgeomNavCollection.hh"
#elif CELER_VGNAV == CELER_VGNAV_TUPLE
#    include <VecGeom/navigation/NavStateTuple.h>
#else
#    include <VecGeom/navigation/NavStateIndex.h>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//

inline constexpr VgSurfaceInt vg_null_surface{-1};
inline constexpr VgNavIndex vg_outside_nav_index{0};

//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
struct VecgeomScalars
{
    using vol_level_uint = VolumeLevelId::size_type;

    VgPlacedVolume<MemSpace::host> const* host_world{nullptr};
    VgPlacedVolume<MemSpace::device> const* device_world{nullptr};
    vol_level_uint num_volume_levels{0};

    template<MemSpace M>
    CELER_FUNCTION VgPlacedVolume<M> const* world() const
    {
        if constexpr (M == MemSpace::host)
        {
            return host_world;
        }
        else if constexpr (M == MemSpace::device)
        {
            return device_world;
        }
#if CELER_CUDACC_BUGGY_IF_CONSTEXPR
        CELER_ASSERT_UNREACHABLE();
#endif
    }

    //! Whether the scalars are valid (device may be null)
    explicit CELER_FUNCTION operator bool() const
    {
        return host_world != nullptr && num_volume_levels > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent data used by VecGeom implementation.
 *
 * The volumes and volume instance mappings are set when constructing from an
 * external model, using VolumeParams to map to Geant4 geometry. For models
 * loaded through VGDML, the mapping is currently one-to-one for implementation
 * and placed volumes.
 */
template<Ownership W, MemSpace M>
struct VecgeomParamsData
{
    using ImplVolInstanceId = VgVolumeInstanceId;

    //! Values that don't require host/device copying
    VecgeomScalars scalars;

    //! Map logical volume ID to canonical
    Collection<VolumeId, W, M, ImplVolumeId> volumes;
    //! Map placed volume ID to canonical
    Collection<VolumeInstanceId, W, M, ImplVolInstanceId> volume_instances;

    //! Whether the data is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        // Volumes/instances can be absent if built from GDML
        return scalars && !volumes.empty() && !volume_instances.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    VecgeomParamsData& operator=(VecgeomParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        scalars = other.scalars;
        volumes = other.volumes;
        volume_instances = other.volume_instances;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Interface for VecGeom state information.
 */
template<Ownership W, MemSpace M>
struct VecgeomStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = StateCollection<T, W, M>;
#if CELER_VGNAV == CELER_VGNAV_PATH
    using VgStateItems = detail::VecgeomNavCollection<W, M>;
#else
    using VgStateItems = StateItems<VgNavStateImpl>;
#endif

    //// DATA ////

    // Physical state
    StateItems<Real3> pos;
    StateItems<Real3> dir;

    // Logical volumetric state
    VgStateItems state;
    StateItems<VgBoundary> boundary;  // Empty if VGNAV=path
    VgStateItems next_state;  // TODO: prev_state
    StateItems<VgBoundary> next_boundary;  // Empty if VGNAV=path

    // Surface state
    StateItems<VgSurfaceInt> next_surf;  // Empty unless using surface model

    //// METHODS ////

    //! True if sizes are consistent and states are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        // clang-format off
        return pos.size() > 0
            && dir.size() == pos.size()
            && state.size() == pos.size()
            && boundary.size() == (CELER_VGNAV != CELER_VGNAV_PATH ? pos.size() : 0)
            && next_state.size() == pos.size()
            && next_boundary.size() == (CELER_VGNAV != CELER_VGNAV_PATH ? pos.size() : 0)
            && next_surf.size() == (CELERITAS_VECGEOM_SURFACE ? pos.size() : 0);
        // clang-format on
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const { return pos.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    VecgeomStateData& operator=(VecgeomStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        pos = other.pos;
        dir = other.dir;
        state = other.state;
        boundary = other.boundary;
        next_state = other.next_state;
        next_boundary = other.next_boundary;
        next_surf = other.next_surf;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// Resize geometry states
template<MemSpace M>
void resize(VecgeomStateData<Ownership::value, M>* data,
            HostCRef<VecgeomParamsData> const& params,
            size_type size);

extern template void
resize<MemSpace::host>(VecgeomStateData<Ownership::value, MemSpace::host>*,
                       HostCRef<VecgeomParamsData> const&,
                       size_type);

extern template void
resize<MemSpace::device>(VecgeomStateData<Ownership::value, MemSpace::device>*,
                         HostCRef<VecgeomParamsData> const&,
                         size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
