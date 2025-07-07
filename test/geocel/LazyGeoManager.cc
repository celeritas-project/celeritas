//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/LazyGeoManager.cc
//---------------------------------------------------------------------------//
#include "LazyGeoManager.hh"

#include <gtest/gtest.h>

#include "corecel/Config.hh"

#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct LazyGeoManager::LazyGeo
{
    std::string key{};
    SPConstGeoI geo{};
};

//---------------------------------------------------------------------------//
class LazyGeoManager::CleanupGeoEnvironment : public ::testing::Environment
{
  public:
    void SetUp() override {}
    void TearDown() override { LazyGeoManager::reset_geometry(); }
};

//---------------------------------------------------------------------------//
auto LazyGeoManager::get_geometry(std::string_view key) -> SPConstGeoI
{
    auto& lazy = LazyGeoManager::lazy_geo();

    if (key != lazy.key)
    {
        // MUST reset geometry before trying to build a new one
        // since VecGeom is all full of globals
        this->reset_geometry();
        lazy.geo = this->build_fresh_geometry(key);
        lazy.key = key;
    }

    return lazy.geo;
}

//---------------------------------------------------------------------------//
/*!
 * Destroy the geometry if needed.
 */
void LazyGeoManager::reset_geometry()
{
    auto& lazy = LazyGeoManager::lazy_geo();
    if (lazy.geo)
    {
        CELER_LOG(debug) << "Destroying '" << lazy.key << "' geometry";
    }
    lazy = {};
}

//---------------------------------------------------------------------------//
auto LazyGeoManager::lazy_geo() -> LazyGeo&
{
    static bool registered_cleanup = false;
    if (!registered_cleanup)
    {
        /*! Always reset geometry at end of testing before global destructors.
         *
         * This is needed because VecGeom stores its objects as static globals,
         * and only makes those objects visible with references/raw data. Thus
         * we can't guarantee that the CoreGeoParams destructor is calling a
         * valid global VecGeom pointer when it destructs, since static
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
