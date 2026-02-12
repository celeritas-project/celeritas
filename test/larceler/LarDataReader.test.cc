//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarDataReader.test.cc
//---------------------------------------------------------------------------//
#include "larceler/io/LarDataReader.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class LarDataReaderTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override
    {
        reader_ = std::make_unique<LarDataReader>(
            this->test_data_path("larceler", "larsim-dune-data.root"));
    }

  protected:
    std::unique_ptr<LarDataReader> reader_;
};

TEST_F(LarDataReaderTest, read)
{
    EXPECT_EQ(10, reader_->num_events());
    EXPECT_EQ("dune10kt_v1_1x2x6", reader_->detector_name());
    EXPECT_EQ(120, reader_->optical_detector_centers().size());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
