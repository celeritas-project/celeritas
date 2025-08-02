//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GlobalGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "GlobalGeoTestBase.hh"

#include <memory>
#include <string>
#include <string_view>

#include "corecel/Config.hh"

#include "geocel/GeantGeoParams.hh"
#include "celeritas/geo/CoreGeoParams.hh"

#include "PersistentSP.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
using PersistentGeantGeo = PersistentSP<GeantGeoParams>;

PersistentGeantGeo& persistent_geant_geo()
{
    static PersistentGeantGeo pgg{"geant4 geometry"};
    return pgg;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
auto GlobalGeoTestBase::build_geometry() -> SPConstCoreGeo
{
    return std::dynamic_pointer_cast<CoreGeoParams const>(
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

    constexpr bool use_orange = CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE;
    constexpr bool use_g4 = CELERITAS_USE_GEANT4;

    // Construct filename:
    // ${SOURCE}/test/celeritas/data/${basename}${fileext}
    auto ext = use_g4 ? ".gdml" : use_orange ? ".org.json" : ".invalid-geo";
    std::string filename
        = this->test_data_path("geocel", std::string{basename} + ext);
    if (use_g4)
    {
        auto& pgg = persistent_geant_geo();

        pgg.clear();
        auto new_geo = GeantGeoParams::from_gdml(filename);
        pgg.set(std::string{basename}, std::move(new_geo));

        // Geant4 loading is supported
        return CoreGeoParams::from_geant(pgg.value());
    }
    else if (use_orange)
    {
        // Using ORANGE without Geant4 (ultralite CI test harness)
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
        return CoreGeoParams::from_json(filename);
#endif
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Access persistent geant geometry after construction.
 */
auto GlobalGeoTestBase::geant_geo() -> SPGeantGeo const&
{
    auto& pgg = persistent_geant_geo();
    return pgg.value();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
