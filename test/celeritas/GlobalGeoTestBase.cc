//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GlobalGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "GlobalGeoTestBase.hh"

#include <memory>
#include <string>
#include <gtest/gtest.h>

#include "celeritas_config.h"
#include "corecel/io/Logger.hh"
#include "celeritas/geo/GeoParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
auto GlobalGeoTestBase::build_geometry() -> SPConstGeo
{
    auto& lazy = GlobalGeoTestBase::lazy_geo();

    // Construct filename
    char const* basename_cstr = this->geometry_basename();
    CELER_ASSERT(basename_cstr);
    std::string basename{basename_cstr};
    CELER_ASSERT(!basename.empty());

    if (basename != lazy.basename)
    {
        // Construct filename:
        // ${SOURCE}/test/celeritas/data/${basename}${fileext}
        char const* ext = CELERITAS_USE_VECGEOM ? ".gdml" : ".org.json";
        auto filename = basename + ext;
        std::string test_file
            = this->test_data_path("celeritas", filename.c_str());

        // MUST reset geometry before trying to build a new one
        // since VecGeom is all full of globals
        GlobalGeoTestBase::reset_geometry();
        lazy.geo = std::make_shared<GeoParams>(test_file.c_str());
        lazy.basename = std::move(basename);
    }

    return lazy.geo;
}

//---------------------------------------------------------------------------//
/*!
 * Destroy the geometry if needed.
 */
void GlobalGeoTestBase::reset_geometry()
{
    auto& lazy = GlobalGeoTestBase::lazy_geo();
    if (lazy.geo)
    {
        CELER_LOG(debug) << "Destroying '" << lazy.basename << "' geometry";
        lazy.geo.reset();
    }
}

//---------------------------------------------------------------------------//
auto GlobalGeoTestBase::lazy_geo() -> LazyGeo&
{
    static bool registered_cleanup = false;
    if (!registered_cleanup)
    {
        /*! Always reset geometry at end of testing before global destructors.
         *
         * This is needed because VecGeom stores its objects as static globals,
         * and only makes those objects visible with references/raw data. Thus
         * we can't guarantee that the GeoParams destructor is calling a valid
         * global VecGeom pointer when it destructs, since static
         * initialization/destruction order is undefined across translation
         * units.
         */
        CELER_LOG(debug) << "Registering CleanupGeoEnvironment";
        ::testing::AddGlobalTestEnvironment(new CleanupGeoEnvironment());
        registered_cleanup = true;
    }

    // Delayed initialization
    static LazyGeo lg;
    return lg;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
