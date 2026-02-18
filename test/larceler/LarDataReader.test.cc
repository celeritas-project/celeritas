//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarDataReader.test.cc
//---------------------------------------------------------------------------//
#include "larceler/LarDataReader.hh"

#include "celeritas/ext/ScopedRootErrorHandler.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class LarDataReaderTest : public ::celeritas::test::Test
{
  protected:
    ScopedRootErrorHandler handle_root_;
};

TEST_F(LarDataReaderTest, read)
{
    LarDataReader reader{
        this->test_data_path("larceler", "larsim-dune-data.root")};
    EXPECT_EQ(10, reader.num_events());
    EXPECT_EQ("dune10kt_v1_1x2x6", reader.detector_name());
    EXPECT_EQ(120, reader.optical_detector_centers().size());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
