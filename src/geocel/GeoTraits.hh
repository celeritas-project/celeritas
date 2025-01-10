//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class GeoParamsInterface;

//---------------------------------------------------------------------------//
/*!
 * Traits class for defining params and device data.
 * \tparam G Geometry params class, e.g. VecgeomParams
 *
 * This traits class \em must be defined for all geometry types. The generic
 * instance here is provided as a synopsis and to improve error checking.
 */
template<class G>
struct GeoTraits
{
    static_assert(std::is_base_of_v<GeoParamsInterface, G>,
                  "G must be a geometry params, not params data");
    static_assert(std::is_void_v<G>, "Geo traits must be specialized");

    //! Params data used during runtime
    template<Ownership W, MemSpace M>
    using ParamsData = void;

    //! State data used during runtime
    template<Ownership W, MemSpace M>
    using StateData = void;

    //! Geometry track view
    using TrackView = void;

    //! Descriptive name for the geometry
    static constexpr char const* name = nullptr;

    //! TO BE REMOVED: "native" file extension for this geometry
    static constexpr char const* ext = nullptr;
};

//---------------------------------------------------------------------------//
//! Helper for determining whether a geometry type is available.
template<class G>
inline constexpr bool is_geometry_configured_v
    = !std::is_void_v<typename GeoTraits<G>::TrackView>;

//---------------------------------------------------------------------------//
/*!
 * Traits class for marking a geometry as not configured.
 *
 * Specializations should inherit from this class when the geometry is not
 * configured.
 */
struct NotConfiguredGeoTraits
{
    template<Ownership W, MemSpace M>
    using ParamsData = void;
    template<Ownership W, MemSpace M>
    using StateData = void;
    using TrackView = void;
    static constexpr char const* name = nullptr;
    static constexpr char const* ext = nullptr;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
