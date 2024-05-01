//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-geo/GeoInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <nlohmann/json.hpp>

#include "corecel/Types.hh"

#include "Types.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Input for setting up the problem for multiple raytraces.
 */
struct ModelSetup
{
    // Global environment
    size_type cuda_stack_size{};
    size_type cuda_heap_size{};

    //! Geometry filename to load (usually GDML)
    std::string geometry_file;
};

//---------------------------------------------------------------------------//
/*!
 * Input for generating a raytrace.
 */
struct TraceSetup
{
    //! Rendering geometry
    Geometry geometry{default_geometry()};

    //! Rendering memory space
    MemSpace memspace{default_memspace()};

    //! Whether to output volume names for this geometry
    bool volumes{false};

    //! Output filename for binary
    std::string bin_file;
};

//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, ModelSetup const& value);
void from_json(nlohmann::json const& j, ModelSetup& value);

void to_json(nlohmann::json& j, TraceSetup const& value);
void from_json(nlohmann::json const& j, TraceSetup& value);

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
