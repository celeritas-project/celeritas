//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomGeoTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "geocel/GeoTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class VecgeomParams;
class VecgeomTrackView;
template<Ownership W, MemSpace M>
struct VecgeomParamsData;
template<Ownership W, MemSpace M>
struct VecgeomStateData;

#if CELERITAS_USE_VECGEOM
//---------------------------------------------------------------------------//
/*!
 * Traits specialization for VecGeom geometry.
 */
template<>
struct GeoTraits<VecgeomParams>
{
    //! Params data used during runtime
    template<Ownership W, MemSpace M>
    using ParamsData = VecgeomParamsData<W, M>;

    //! State data used during runtime
    template<Ownership W, MemSpace M>
    using StateData = VecgeomStateData<W, M>;

    //! Geometry track view
    using TrackView = VecgeomTrackView;

    //! Surfaces are available when using frame implementation
    static constexpr bool has_impl_surface{CELERITAS_VECGEOM_SURFACE};
    //! VecGeom has "placed volumes"
    static constexpr bool has_impl_volume_instance = true;

    //! Descriptive name for the geometry
    static constexpr char const name[] = "VecGeom";
};
#else
//! VecGeom is unavailable
template<>
struct GeoTraits<VecgeomParams> : NotConfiguredGeoTraits
{
};
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
