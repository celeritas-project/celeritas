//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoParamsTest.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_test.hh"

#include "geometry/GeoParams.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//

class GeoParamsTest : public celeritas::Test
{
  protected:
    using SPConstGeo = std::shared_ptr<const celeritas::GeoParams>;

    static void SetUpTestCase()
    {
        std::string test_file
            = celeritas::Test::test_data_path("geometry", "twoBoxes.gdml");
        geom_ = std::make_shared<celeritas::GeoParams>(test_file.c_str());
    }

    static void TearDownTestCase() { geom_.reset(); }

    const SPConstGeo& params()
    {
        CELER_ENSURE(geom_);
        return geom_;
    }

  private:
    static SPConstGeo geom_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
