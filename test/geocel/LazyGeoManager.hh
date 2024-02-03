//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/LazyGeoManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string_view>

#include "geocel/GeoParamsInterface.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Base class for managing a persistent singleton geometry.
 *
 * This class automatically cleans up after all tests are done executing, and
 * ensures that only one geometry at a time is loaded.
 */
class LazyGeoManager
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstGeoI = std::shared_ptr<GeoParamsInterface const>;
    //!@}

  public:
    //! Construct a geometry for the first time
    virtual SPConstGeoI build_fresh_geometry(std::string_view key) = 0;

    // Construct or access the geometry
    SPConstGeoI get_geometry(std::string_view key);

    // Clear the lazy geometry
    static void reset_geometry();

  private:
    struct LazyGeo;
    class CleanupGeoEnvironment;
    static LazyGeo& lazy_geo();
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
