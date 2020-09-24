//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoParamsTest.hh
//---------------------------------------------------------------------------//
#pragma once

#include "gtest/Main.hh"
#include "gtest/Test.hh"

#include "geometry/GeoParams.hh"

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

namespace celeritas_test
{
using namespace celeritas;

class GeoParamsTest : public celeritas::Test
{
  protected:
    using SptrConstParams = std::shared_ptr<const GeoParams>;

    static void SetUpTestCase()
    {
        std::string test_file
            = celeritas::Test::test_data_path("geometry", "twoBoxes.gdml");
        geom_ = std::make_shared<GeoParams>(test_file.c_str());
    }

    static void TearDownTestCase() { geom_.reset(); }

    const SptrConstParams& params()
    {
        ENSURE(geom_);
        return geom_;
    }

  private:
    static SptrConstParams geom_;
};

} // namespace celeritas_test
