//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/LazyGeantGeoManager.cc
//---------------------------------------------------------------------------//
#include "LazyGeantGeoManager.hh"

#include "corecel/io/StringUtils.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"

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
using PersistentVolumes = PersistentSP<VolumeParams const>;

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

PersistentVolumes& persistent_volumes()
{
    static PersistentVolumes pv{"volumes"};
    return pv;
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
    std::string const basename{this->gdml_basename()};
    CELER_VALIDATE(!basename.empty(), << "invalid basename");

    auto& pgeo = persistent_geo();
    pgeo.lazy_update(basename, [this, &basename]() {
        // Get GDML filename
        std::string filename = [&basename] {
            if (starts_with(basename, "/"))
            {
                // Absolute path: use this filename
                return basename;
            }
            // ${SOURCE}/test/celeritas/data/${basename}.gdml
            return Test::test_data_path("geocel", basename + ".gdml");
        }();

        SPConstGeoI new_geo;
        if constexpr (CELERITAS_USE_GEANT4)
        {
            auto& pgeant_geo = persistent_geant_geo();
            // This is called *unless* the user has manually cleared the
            // secondary geometry and reloads from the same Geant4 geo
            pgeant_geo.lazy_update(basename, [&]() {
                auto result = this->build_geant_geo(filename);
                auto volumes = std::make_shared<VolumeParams const>(
                    result->make_model_input().volumes);
                persistent_volumes().set(basename, std::move(volumes));
                return result;
            });

            // Build specific geometry
            new_geo = this->build_geo_from_geant(pgeant_geo.value());
        }
        else
        {
            // Fallback: geometry may be able to build without Geant4
            new_geo = this->build_geo_from_gdml(filename);
            auto volumes = std::make_shared<VolumeParams const>(
                new_geo->make_model_input().volumes);
            persistent_volumes().set(basename, std::move(volumes));
        }

        CELER_ASSERT(new_geo);
        return new_geo;
    });

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
/*!
 * Access volumes from built geometry or geant4 model.
 */
auto LazyGeantGeoManager::volumes() const -> SPConstVolumes
{
    auto& pgg = persistent_volumes();
    if (pgg.key() == this->gdml_basename())
    {
        return pgg.value();
    }
    return nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Reset the secondary geometry manually.
 *
 * This is needed by AllGeoTypedTestBase, where multiple versions of the same
 * Geant4 geometry are loaded in a single test.
 */
void LazyGeantGeoManager::clear_lazy_geo()
{
    persistent_geo().clear();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
