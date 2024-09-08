//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GlobalGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "GlobalGeoTestBase.hh"

#include <memory>
#include <string>
#include <string_view>

#include "corecel/Config.hh"

#include "celeritas/geo/GeoParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
auto GlobalGeoTestBase::build_geometry() -> SPConstGeo
{
    return std::dynamic_pointer_cast<GeoParams const>(
        this->get_geometry(this->geometry_basename()));
}

//---------------------------------------------------------------------------//
/*!
 * Construct a new geometry.
 *
 * This only occurs after any existing built geometries have been cleared. The
 * argument is the geometry basename.
 */
auto GlobalGeoTestBase::build_fresh_geometry(std::string_view basename)
    -> SPConstGeoI
{
    using namespace std::literals;

    // Construct filename:
    // ${SOURCE}/test/celeritas/data/${basename}${fileext}
    auto ext = ".gdml"sv;
    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
        && (!CELERITAS_USE_GEANT4
            || CELERITAS_REAL_TYPE != CELERITAS_REAL_TYPE_DOUBLE))
    {
        // Using ORANGE, either without Geant4 or without double-precision
        // arithmetic
        ext = ".org.json"sv;
    }
    auto filename = std::string{basename} + std::string{ext};
    std::string test_file = this->test_data_path("geocel", filename);
    return std::make_shared<GeoParams>(test_file);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
