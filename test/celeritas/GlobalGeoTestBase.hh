//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GlobalGeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string_view>

#include "geocel/LazyGeoManager.hh"

#include "GlobalTestBase.hh"

namespace celeritas
{
class GeantGeoParams;
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Reuse geometry across individual tests.
 *
 * This is helpful for slow geometry construction or if the geometry has
 * trouble building/destroying multiple times per execution due to global
 * variable usage (VecGeom, Geant4).
 *
 * The "geometry basename" should be the filename without extension of a
 * geometry file inside `test/celeritas/data`.
 */
class GlobalGeoTestBase : virtual public GlobalTestBase,
                          protected LazyGeoManager
{
  public:
    // Overload with the base filename of the geometry
    virtual std::string_view geometry_basename() const = 0;

    // Construct a geometry that's persistent across tests
    SPConstCoreGeo build_geometry() override;

  protected:
    using SPGeantGeo = std::shared_ptr<GeantGeoParams>;

    // Access persistent geant geometry after construction
    static SPGeantGeo const& geant_geo();

    // Lazy geometry construction and cleanup
    SPConstGeoI build_fresh_geometry(std::string_view) override;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
