//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SDParams.test.cc
//------------

#include "celeritas/user/SDParams.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "celeritas/GlobalGeoTestBase.hh"
#include "celeritas/OnlyCoreTestBase.hh"
#include "celeritas/OnlyGeoTestBase.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"
#include "gtest/gtest.h"

namespace celeritas
{

namespace test
{
class SDParamsTest : public GlobalGeoTestBase,
                     public OnlyGeoTestBase,
                     public OnlyCoreTestBase
{
  public:
    using VecLabel = std::vector<Label>;
    std::string_view geometry_basename() const override
    {
        return "testem3-flat";
    }

  protected:
};

TEST_F(SDParamsTest, detector_test) {}
}  // namespace test
}  // namespace celeritas