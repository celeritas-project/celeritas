//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SDParams.test.cc
//------------

#include "celeritas/user/SDParams.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "celeritas/geo/HeuristicGeoTestBase.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"
#include "gtest/gtest.h"

namespace celeritas
{

namespace test
{
class SDParamsTest : public HeuristicGeoTestBase
{
  public:
    using VecLabel = std::vector<Label>;
    std::string_view geometry_basename() const override
    {
        return "testem3-flat";
    }

    //! Construct problem-specific attributes (sampling box etc)
    HeuristicGeoScalars build_scalars() const { return {}; }
    //! Build a list of volumes to compare average paths
    SpanConstStr reference_volumes() const { return {}; }
    //! Return the vector of path lengths mapped by sorted volume name
    SpanConstReal reference_avg_path() const { return {}; }

  protected:
};

TEST_F(SDParamsTest, detector_test) {}
}  // namespace test
}  // namespace celeritas