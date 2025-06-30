//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeGeoTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/GeoTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class OrangeParams;
class OrangeTrackView;
template<Ownership W, MemSpace M>
struct OrangeParamsData;
template<Ownership W, MemSpace M>
struct OrangeStateData;

//---------------------------------------------------------------------------//
/*!
 * Traits specialization for ORANGE geometry.
 */
template<>
struct GeoTraits<OrangeParams>
{
    //! Params data used during runtime
    template<Ownership W, MemSpace M>
    using ParamsData = OrangeParamsData<W, M>;

    //! State data used during runtime
    template<Ownership W, MemSpace M>
    using StateData = OrangeStateData<W, M>;

    //! Geometry track view
    using TrackView = OrangeTrackView;

    //! ORANGE implementation uses surfaces
    static constexpr bool has_impl_surface = true;

    //! Descriptive name for the geometry
    static constexpr char const name[] = "ORANGE";

    //! TO BE REMOVED: "native" file extension for this geometry
    static constexpr char const ext[] = ".org.json";
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
