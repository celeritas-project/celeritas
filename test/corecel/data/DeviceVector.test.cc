//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/DeviceVector.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/DeviceVector.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(DeviceVectorTest, all)
{
    using Vec_t = DeviceVector<int>;
    Vec_t vec;
    EXPECT_EQ(0, vec.size());
    EXPECT_TRUE(vec.empty());

    if (!CELER_USE_DEVICE)
    {
        // Test that allocation fails
        EXPECT_THROW(Vec_t(1234), DebugError);
        return;
    }

    vec = Vec_t(1024);
    EXPECT_EQ(1024, vec.size());
    {
        Vec_t other(128);
        int* orig_vec = vec.device_ref().data();
        int* orig_other = other.device_ref().data();
        swap(vec, other);
        EXPECT_EQ(1024, other.size());
        EXPECT_EQ(orig_other, vec.device_ref().data());
        EXPECT_EQ(orig_vec, other.device_ref().data());
    }
    EXPECT_EQ(128, vec.size());

    std::vector<int> data(vec.size());
    data.front() = 1;
    data.back() = 1234567;

    vec.copy_to_device(make_span(data));

    std::vector<int> newdata(vec.size());
    vec.copy_to_host(make_span(newdata));
    EXPECT_EQ(1, newdata.front());
    EXPECT_EQ(1234567, newdata.back());

    // Test move construction/assignment
    {
        int* orig_vec = vec.device_ref().data();
        Vec_t other(std::move(vec));
        EXPECT_EQ(128, other.size());
        EXPECT_EQ(0, vec.size());
        EXPECT_EQ(orig_vec, other.device_ref().data());
    }
}

/*!
 * The following test code is intentionally commented out. Define
 * CELERITAS_SHOULD_NOT_COMPILE to check that the enclosed code results in
 * the expected build errors.
 */
#ifdef CELERITAS_SHOULD_NOT_COMPILE
TEST(DeviceVectorTest, should_not_compile)
{
    DeviceVector<int> dv(123);
    Span<int> s = make_span(dv);
    EXPECT_EQ(123, s.size());

    auto const& dv_cref = dv;
    Span<int const> s2 = make_span(dv_cref);
    EXPECT_EQ(123, s2.size());
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
