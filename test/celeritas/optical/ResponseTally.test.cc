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

    //! Combine start and end bins, returning -1 for a null result
    long long bin(size_type det, Real3 const& pos, real_type time) const
    {
        ResponseTallyView view(params_.host_ref());
        auto result = view.end_bin(view.start_bin(pos), DetectorId{det}, time);
        return result ? static_cast<long long>(result.get()) : -1;
    }

    ParamsDataStore<ResponseTallyParamsData> params_;
};

TEST_F(ResponseTallyTest, params)
{
    auto const& ref = params_.host_ref();
    EXPECT_EQ(2 * 2 * 3 * 4 * 5, ref.num_bins());
}

TEST_F(ResponseTallyTest, start_bin)
{
    ResponseTallyView view(params_.host_ref());
    EXPECT_EQ(240u, view.num_bins());

    // Only the spatial coordinates contribute to the partial bin
    EXPECT_EQ(0u, view.start_bin({0.5, 0.5, 0.5}).get());
    // z has stride 5, y stride 20, x stride 60
    EXPECT_EQ(5u, view.start_bin({0.5, 0.5, 1.5}).get());
    EXPECT_EQ(20u, view.start_bin({0.5, 1.5, 0.5}).get());
    EXPECT_EQ(60u, view.start_bin({1.5, 0.5, 0.5}).get());
    EXPECT_EQ(115u, view.start_bin({1.5, 2.5, 3.5}).get());

    // Outside the spatial grid
    EXPECT_FALSE(view.start_bin({-0.1, 0.5, 0.5}));
    EXPECT_FALSE(view.start_bin({2.0, 0.5, 0.5}));
    EXPECT_FALSE(view.start_bin({0.5, 3.5, 0.5}));
    EXPECT_FALSE(view.start_bin({0.5, 0.5, 4.0}));
}

TEST_F(ResponseTallyTest, end_bin)
{
    // First bin
    EXPECT_EQ(0, this->bin(0, {0.5, 0.5, 0.5}, 0.5));
    // Time changes fastest
    EXPECT_EQ(1, this->bin(0, {0.5, 0.5, 0.5}, 2.5));
    EXPECT_EQ(4, this->bin(0, {0.5, 0.5, 0.5}, 9.5));
    // Detector is slowest (stride 120)
    EXPECT_EQ(120, this->bin(1, {0.5, 0.5, 0.5}, 0.5));
    // Last bin
    EXPECT_EQ(239, this->bin(1, {1.5, 2.5, 3.5}, 9.5));
    // Contributions add
    EXPECT_EQ(120 + 60 + 20 + 5 + 3, this->bin(1, {1.5, 1.5, 1.5}, 6.5));
}

TEST_F(ResponseTallyTest, out_of_range)
{
    ResponseTallyView view(params_.host_ref());

    // Bad start
    EXPECT_EQ(-1, this->bin(0, {-0.1, 0.5, 0.5}, 0.5));
    EXPECT_EQ(-1, this->bin(0, {0.5, 0.5, 4.0}, 0.5));
    // Bad detector
    EXPECT_EQ(-1, this->bin(2, {0.5, 0.5, 0.5}, 0.5));
    // Bad time
    EXPECT_EQ(-1, this->bin(0, {0.5, 0.5, 0.5}, -1));
    EXPECT_EQ(-1, this->bin(0, {0.5, 0.5, 0.5}, 10));

    // A null starting bin is never scored
    EXPECT_FALSE(view.end_bin(ResponseBinId{}, DetectorId{0}, 0.5));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
