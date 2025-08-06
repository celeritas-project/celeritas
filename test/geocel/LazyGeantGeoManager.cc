//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/LazyGeantGeoManager.cc
//---------------------------------------------------------------------------//
#include "LazyGeantGeoManager.hh"

#include "corecel/io/StringUtils.hh"
#include "geocel/GeantGeoParams.hh"

#include "PersistentSP.hh"
#include "Test.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
using PersistentGeoI = PersistentSP<GeoParamsInterface const>;
using PersistentGeantGeo = PersistentSP<GeantGeoParams const>;

PersistentGeoI& persistent_geo()
{
    static PersistentGeoI pgi{"geometry"};
    return pgi;
}

PersistentGeantGeo& persistent_geant_geo()
{
    static PersistentGeantGeo pgg{"geant4 geometry"};
    return pgg;
}

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Access the basename of the geometry that's currently cached.
 */
std::string const& LazyGeantGeoManager::cached_gdml_basename() const
{
    return persistent_geo().key();
}

//---------------------------------------------------------------------------//
/*!
 * Construct a geometry for the first time.
 */
auto LazyGeantGeoManager::lazy_geo() const -> SPConstGeoI
{
    std::string basename{this->gdml_basename()};
    CELER_VALIDATE(!basename.empty(), << "invalid basename");

    auto& pgeo = persistent_geo();
    if (basename != pgeo.key())
    {
        // Reset secondary geometry, then Geant4 geometry
        pgeo.clear();

        // ${SOURCE}/test/celeritas/data/${basename}.gdml
        std::string filename = [&basename] {
            if (starts_with(basename, "/"))
            {
                // Absolute path: use this filename
                return basename;
            }
            return Test::test_data_path("geocel", basename + ".gdml");
        }();

        if constexpr (CELERITAS_USE_GEANT4)
        {
            auto& pgeant_geo = persistent_geant_geo();
            pgeant_geo.clear();

            // Load geant4 geometry
            auto new_geant_geo = this->build_geant_geo(filename);
            pgeant_geo.set(basename, new_geant_geo);

            // Build specific geometry
            auto new_geo = this->build_geo_from_geant(new_geant_geo);
            CELER_ASSERT(new_geo);
            pgeo.set(basename, std::move(new_geo));
        }
        else
        {
            // Fallback: geometry may be able to build without Geant4
            auto new_geo = this->build_geo_from_gdml(filename);
            CELER_ASSERT(new_geo);
            pgeo.set(basename, std::move(new_geo));
        }
    }

    CELER_ENSURE(pgeo.value());
    return pgeo.value();
}

//---------------------------------------------------------------------------//
/*!
 * Build a Geant4 geometry.
 */
auto LazyGeantGeoManager::build_geant_geo(std::string const& filename) const
    -> SPConstGeantGeo
{
    return GeantGeoParams::from_gdml(filename);
}

//---------------------------------------------------------------------------//
/*!
 * Build from a GDML path as a fallback.
 */
auto LazyGeantGeoManager::build_geo_from_gdml(std::string const&) const
    -> SPConstGeoI
{
    if constexpr (CELERITAS_USE_GEANT4)
    {
        CELER_ASSERT_UNREACHABLE();
    }

    CELER_NOT_IMPLEMENTED("constructing geometry without Geant4 enabled");
}
//---------------------------------------------------------------------------//
/*!
 * Access persistent geant geometry after construction.
 */
auto LazyGeantGeoManager::geant_geo() const -> SPConstGeantGeo
{
    auto& pgg = persistent_geant_geo();
    if (pgg.key() == this->gdml_basename())
    {
        return pgg.value();
    }
    return nullptr;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
