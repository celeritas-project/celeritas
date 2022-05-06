//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geometry/GeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <utility>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

#include "gtest/Test.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Test class that lazily loads an input geometry.
 *
 * This should be instantiated on geometry type.
 */
template<class GP>
class GeoTestBase : public celeritas::Test
{
  public:
    //!@{
    //! Type aliases
    using GeoParams  = GP;
    using SPConstGeo = std::shared_ptr<const GeoParams>;
    //!@}

  public:
    //! Default parent directory of test data
    virtual const char* dirname() const = 0;

    //! Daughter class must set up a geometry name
    virtual const char* filebase() const = 0;

    //! Default file extension
    virtual const char* fileext() const
    {
        return CELERITAS_USE_VECGEOM ? ".gdml" : ".org.json";
    }

    // Lazy load of geometry
    const SPConstGeo& geometry() const;

    // Always reset geometry at end of test suite
    static void TearDownTestCase();

  private:
    struct LazyGeo
    {
        std::string filebase{};
        SPConstGeo  geo{};
    };

    static LazyGeo lazy_;
};

//---------------------------------------------------------------------------//
// TEMPLATE CLASSS DEFINITIONS
//---------------------------------------------------------------------------//

template<class GP>
typename GeoTestBase<GP>::LazyGeo GeoTestBase<GP>::lazy_;

//---------------------------------------------------------------------------//
/*!
 * Always reset geometry at end of test suite.
 *
 * This is needed because VecGeom stores its objects as static globals, and
 * only makes those objects visible with references/raw data. Thus we can't
 * guarantee that the GeoParams destructor is calling a valid global VecGeom
 * pointer when it destructs, since static initialization/destruction order is
 * undefined across translation units.
 */
template<class GP>
void GeoTestBase<GP>::TearDownTestCase()
{
    if (lazy_.geo)
    {
        lazy_.geo.reset();
        lazy_.filebase.clear();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Lazily load geometry.
 */
template<class GP>
auto GeoTestBase<GP>::geometry() const -> const SPConstGeo&
{
    std::string filebase = this->filebase();
    CELER_ASSERT(!filebase.empty());
    if (lazy_.filebase != filebase)
    {
        // Construct filename:
        // ${SOURCE}/test/${dirname}/data/${filebase}${fileext}
        auto        filename  = filebase + this->fileext();
        std::string test_file = celeritas::Test::test_data_path(
            this->dirname(), filename.c_str());

        // MUST reset geometry before trying to build a new one
        // since VecGeom is all full of globals
        lazy_.geo.reset();
        lazy_.geo      = std::make_shared<GeoParams>(test_file.c_str());
        lazy_.filebase = std::move(filebase);
    }

    CELER_ENSURE(lazy_.geo);
    return lazy_.geo;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
