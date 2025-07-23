//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/LazyGeoManager.cc
//---------------------------------------------------------------------------//
#include "LazyGeoManager.hh"

#include "PersistentSP.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
using PersistentGeo = PersistentSP<GeoParamsInterface const>;

PersistentGeo& persistent_geo()
{
    static PersistentGeo pg{"geometry"};
    return pg;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
auto LazyGeoManager::get_geometry(std::string_view key) -> SPConstGeoI
{
    auto& pg = persistent_geo();

    if (key != pg.key())
    {
        // MUST reset geometry before trying to build a new one
        // since VecGeom is all full of globals
        pg.clear();
        auto new_geo = this->build_fresh_geometry(key);
        CELER_ASSERT(new_geo);
        pg.set(std::string{key}, std::move(new_geo));
    }

    CELER_ENSURE(pg.value());
    return pg.value();
}

//---------------------------------------------------------------------------//
/*!
 * Destroy the geometry if needed.
 */
void LazyGeoManager::reset_geometry()
{
    persistent_geo().clear();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
