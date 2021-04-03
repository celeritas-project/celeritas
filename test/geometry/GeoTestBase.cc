//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoTestBase.cc
//---------------------------------------------------------------------------//
#include "GeoTestBase.hh"

#include "comm/Logger.hh"
#include "geometry/GeoParams.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// Static data
GeoTestBase::LazyGeo GeoTestBase::lazy_;

//---------------------------------------------------------------------------//
/*!
 * Always reset geometry at end of test suite.
 *
 * This is needed because VecGeom stores its objects as static globals, and
 * only makes those objects visible with references/raw pointers. Thus we can't
 * guarantee that the GeoParams destructor is calling a valid global VecGeom
 * pointer when it destructs, since static initialization/destruction order is
 * undefined across translation units.
 */
void GeoTestBase::TearDownTestCase()
{
    if (lazy_.geo)
    {
        CELER_LOG(debug) << "Resetting geometry";
        lazy_.geo.reset();
        lazy_.filename.clear();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Lazily load geometry.
 */
auto GeoTestBase::geo_params() const -> const SPConstGeo&
{
    std::string filename = this->filename();
    CELER_ASSERT(!filename.empty());
    if (lazy_.filename != filename)
    {
        std::string test_file
            = celeritas::Test::test_data_path("geometry", filename.c_str());
        // MUST reset geometry before trying to build a new one
        // since VecGeom is all full of globals
        lazy_.geo.reset();
        lazy_.geo = std::make_shared<celeritas::GeoParams>(test_file.c_str());
        lazy_.filename = std::move(filename);
    }

    CELER_ENSURE(lazy_.geo);
    return lazy_.geo;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
