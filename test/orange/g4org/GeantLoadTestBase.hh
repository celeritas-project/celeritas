//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/GeantLoadTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "geocel/GeantGeoParams.hh"
#include "geocel/LazyGeantGeoManager.hh"

#include "Test.hh"

namespace celeritas
{
namespace g4org
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Load a Geant4 geometry and clean up as needed.
 */
class GeantLoadTestBase : public ::celeritas::test::Test,
                          public ::celeritas::test::LazyGeantGeoManager
{
  public:
    //! Get filename or relative path
    std::string_view gdml_basename() const
    {
        CELER_VALIDATE(!filename_.empty(), << "load_gdml was not called");
        return filename_;
    }

    //! Only geant4 construction is done
    SPConstGeoI build_geo_from_geant(SPConstGeantGeo const& g) const final
    {
        return g;
    }

    // Build via Geant4 GDML reader
    void load_gdml(std::string const& filename)
    {
        filename_ = filename;
        this->lazy_geo();
    }

    // Load a test input
    void load_test_gdml(std::string_view basename)
    {
        filename_ = basename;
        this->lazy_geo();
    }

    //! Access the geo params after loading
    GeantGeoParams const& geo() const
    {
        auto geo = this->geant_geo();
        CELER_ENSURE(geo);
        return *geo;
    }

  private:
    std::string filename_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
