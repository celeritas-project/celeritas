//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "gtest/Test.hh"

#include <string>

namespace celeritas
{
class GeoParams;
}

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Test class for geometry that lazily loads an input file.
 */
class GeoTestBase : public celeritas::Test
{
  public:
    using SPConstGeo = std::shared_ptr<const celeritas::GeoParams>;

    //! Daughter class must set up a geometry name
    virtual std::string filename() const = 0;

    //! Lazy load of geometry
    const SPConstGeo& geo_params() const;

    // Always reset geometry at end of test suite
    static void TearDownTestCase();

  private:
    struct LazyGeo
    {
        std::string filename{};
        SPConstGeo  geo{};
    };

    static LazyGeo lazy_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
