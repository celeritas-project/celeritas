//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
 *
 * This traits class \em must be defined for all geometry types. The generic
 * instance here is provided as a synopsis and to improve error checking.
 */
template<class G>
struct GeoTraits
{
    static_assert(std::is_base_of_v<GeoParamsInterface, G>,
                  "G must be a geometry params, not params data");
    static_assert(sizeof(G) == 0, "Geo traits must be specialized");

    //! Params data used during runtime
    template<Ownership W, MemSpace M>
    using ParamsData = void;

    //! State data used during runtime
    template<Ownership W, MemSpace M>
    using StateData = void;

    //! Geometry track view
    using TrackView = void;

    //! Descriptive name for the geometry
    static inline char const* name = nullptr;

    //! TO BE REMOVED: "native" file extension for this geometry
    static inline char const* ext = nullptr;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
