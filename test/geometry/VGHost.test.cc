//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGHost.test.cc
//---------------------------------------------------------------------------//
#include "geometry/VGHost.hh"

#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "celeritas_config.h"

using celeritas::VGHost;
using VolumeId = celeritas::VolumeId;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class VGHostTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(VGHostTest, all)
{
    std::string test_file = std::string(CELERITAS_SOURCE_DIR)
                            + "/test/geometry/data/twoBoxes.gdml";
    VGHost host(test_file.c_str());

    EXPECT_EQ(2, host.num_volumes());
    EXPECT_EQ(2, host.max_depth());
    EXPECT_EQ("Detector", host.id_to_label(VolumeId{0}));
    EXPECT_EQ("World", host.id_to_label(VolumeId{1}));
}
