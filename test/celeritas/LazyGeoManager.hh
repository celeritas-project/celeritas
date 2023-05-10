//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/LazyGeoManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string_view>

#include "orange/GeoParamsInterface.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Templated base class for creating a persistent geometry.
 *
 * \tparam GP Host Geometry Params class
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
