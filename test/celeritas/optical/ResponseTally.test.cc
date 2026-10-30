//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ResponseTally.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/ParamsDataStore.hh"
#include "corecel/grid/UniformGridData.hh"
#include "celeritas/optical/ResponseTallyData.hh"
#include "celeritas/optical/ResponseTallyView.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//
class ResponseTallyTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override
    {
        // 2 detectors, 2x3x4 spatial bins over [0, 2) x [0, 3) x [0, 4),
        // 5 time bins over [0, 10)
        HostVal<ResponseTallyParamsData> host;
        host.x_grid = UniformGridData::from_bounds({0, 2}, 3);
        host.y_grid = UniformGridData::from_bounds({0, 3}, 4);
        host.z_grid = UniformGridData::from_bounds({0, 4}, 5);
        host.time_grid = UniformGridData::from_bounds({0, 10}, 6);
        host.dims = {2, 2, 3, 4, 5};
        ASSERT_TRUE(host);
        params_ = ParamsDataStore<ResponseTallyParamsData>{std::move(host)};
    }

    ParamsDataStore<ResponseTallyParamsData> params_;
};

TEST_F(ResponseTallyTest, params)
{
    auto const& ref = params_.host_ref();
    EXPECT_EQ(2 * 2 * 3 * 4 * 5, ref.num_bins());
}

TEST_F(ResponseTallyTest, bin_id)
{
    ResponseTallyView view(params_.host_ref());
    EXPECT_EQ(240u, view.num_bins());

    // First bin
    EXPECT_EQ(0u, view.bin_id(0, {0.5, 0.5, 0.5}, 0.5));
    // Time changes fastest
    EXPECT_EQ(1u, view.bin_id(0, {0.5, 0.5, 0.5}, 2.5));
    // Then z (stride 5)
    EXPECT_EQ(5u, view.bin_id(0, {0.5, 0.5, 1.5}, 0.5));
    // Then y (stride 20)
    EXPECT_EQ(20u, view.bin_id(0, {0.5, 1.5, 0.5}, 0.5));
    // Then x (stride 60)
    EXPECT_EQ(60u, view.bin_id(0, {1.5, 0.5, 0.5}, 0.5));
    // Detector is slowest (stride 120)
    EXPECT_EQ(120u, view.bin_id(1, {0.5, 0.5, 0.5}, 0.5));
    // Last bin
    EXPECT_EQ(239u, view.bin_id(1, {1.5, 2.5, 3.5}, 9.5));
}

TEST_F(ResponseTallyTest, out_of_range)
{
    ResponseTallyView view(params_.host_ref());
    auto const invalid = ResponseTallyView::invalid_bin();

    EXPECT_EQ(invalid, view.bin_id(2, {0.5, 0.5, 0.5}, 0.5));
    EXPECT_EQ(invalid, view.bin_id(0, {-0.1, 0.5, 0.5}, 0.5));
    EXPECT_EQ(invalid, view.bin_id(0, {2.0, 0.5, 0.5}, 0.5));
    EXPECT_EQ(invalid, view.bin_id(0, {0.5, 3.5, 0.5}, 0.5));
    EXPECT_EQ(invalid, view.bin_id(0, {0.5, 0.5, 4.0}, 0.5));
    EXPECT_EQ(invalid, view.bin_id(0, {0.5, 0.5, 0.5}, -1));
    EXPECT_EQ(invalid, view.bin_id(0, {0.5, 0.5, 0.5}, 10));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
