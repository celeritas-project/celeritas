//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/ObserverPtr.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/ObserverPtr.hh"

#include <algorithm>
#include <iterator>
#include <type_traits>

#include "corecel/data/DeviceVector.hh"

#include "ObserverPtr.test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class ObserverPtrTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(ObserverPtrTest, host)
{
    using VecInt = std::vector<int>;
    VecInt some_data = {1, 2, 3, 4};

    auto obs = make_observer(some_data.data());
    EXPECT_TRUE(
        (std::is_same_v<decltype(obs), ObserverPtr<int, MemSpace::host>>));
    EXPECT_TRUE(obs);
    EXPECT_TRUE(obs == obs);
    EXPECT_TRUE(obs != nullptr);
    EXPECT_EQ(1, *obs);
    EXPECT_EQ(1, *obs.get());
    EXPECT_EQ(1, *obs.release());
    EXPECT_EQ(nullptr, obs.get());
    EXPECT_TRUE(obs == nullptr);

    auto vec_ptr = make_observer(const_cast<VecInt const*>(&some_data));
    EXPECT_TRUE((std::is_same_v<decltype(vec_ptr),
                                ObserverPtr<VecInt const, MemSpace::host>>));
    EXPECT_EQ(4, vec_ptr->size());
}

TEST_F(ObserverPtrTest, TEST_IF_CELER_DEVICE(device))
{
    int const test_data[] = {1, 1, 2, 3, 5, 8};
    DeviceVector<int> inp_dv(std::size(test_data));
    DeviceVector<int> out_dv(std::size(test_data));
    std::vector<int> out(std::size(test_data), -1);

    auto src = make_observer(inp_dv);
    auto dst = make_observer(out_dv);
#ifdef CELERITAS_SHOULD_NOT_COMPILE
    // Assigning a device pointer on host is prohibited
    *src = 2;
#endif

    // Test device pointer with manual kernel
    inp_dv.copy_to_device(make_span(test_data));
    copy_test(src, dst, inp_dv.size());
    out_dv.copy_to_host(make_span(out));
    EXPECT_VEC_EQ(make_span(test_data), out);

    // (clear data)
    std::fill(out.begin(), out.end(), -1);
    out_dv.copy_to_device(make_span(out));

    // Test device pointer with thrust kernel
    copy_thrust_test(src, dst, inp_dv.size());
    out_dv.copy_to_host(make_span(out));
    EXPECT_VEC_EQ(make_span(test_data), out);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
