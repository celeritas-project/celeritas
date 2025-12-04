//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomData.cc
//---------------------------------------------------------------------------//
#include "VecgeomData.hh"

#include "corecel/Types.hh"
#include "geocel/vg/VecgeomTypes.hh"

#if CELER_VGNAV == CELER_VGNAV_TUPLE
#    include <VecGeom/navigation/NavStateTuple.h>
#else
#    include <VecGeom/navigation/NavStateIndex.h>
#endif

#include "corecel/data/CollectionBuilder.hh"

#include "detail/VecgeomSetup.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * Resize geometry states.
 *
 * \todo Add stream ID
 */
template<MemSpace M>
void resize(VecgeomStateData<Ownership::value, M>* data,
            HostCRef<VecgeomParamsData> const& params,
            size_type size)
{
    CELER_EXPECT(data);
    CELER_EXPECT(size > 0);
    CELER_EXPECT(params);

    resize(&data->pos, size);
    resize(&data->dir, size);
    resize(&data->state, size);
    resize(&data->next_state, size);
    if constexpr (M == MemSpace::device)
    {
#if CELER_VGNAV == CELER_VGNAV_TUPLE
        using AllStates = AllItems<VgNavStateImpl, MemSpace::device>;
        detail::init_navstate_device(data->state[AllStates{}], StreamId{});
        detail::init_navstate_device(data->next_state[AllStates{}], StreamId{});
#endif
    }

    if constexpr (CELER_VGNAV != CELER_VGNAV_PATH)
    {
        // Unless using the 'path' navigator, boundary data is stored
        // independently
        resize(&data->boundary, size);
    }
    if constexpr (CELER_VGNAV != CELER_VGNAV_PATH)
    {
        // Path navigator stores the boundary, and surface model uses next_surf
        resize(&data->next_boundary, size);
    }
    if constexpr (CELERITAS_VECGEOM_SURFACE)
    {
        resize(&data->next_surf, size);
    }

    CELER_ENSURE(data);
}

//---------------------------------------------------------------------------//
template void
resize<MemSpace::host>(VecgeomStateData<Ownership::value, MemSpace::host>*,
                       HostCRef<VecgeomParamsData> const&,
                       size_type);

template void
resize<MemSpace::device>(VecgeomStateData<Ownership::value, MemSpace::device>*,
                         HostCRef<VecgeomParamsData> const&,
                         size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
