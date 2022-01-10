//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceVector.test.cc
//---------------------------------------------------------------------------//
#include "base/DeviceVector.hh"

#include "celeritas_test.hh"

using celeritas::DeviceVector;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class DeviceVectorTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(DeviceVectorTest, all)
{
    using Vec_t = DeviceVector<int>;
    Vec_t vec;
    EXPECT_EQ(0, vec.size());
    EXPECT_TRUE(vec.empty());

#if !CELERITAS_USE_CUDA
    // Can't allocate
    EXPECT_THROW(Vec_t(1234), celeritas::DebugError);
    cout << "CUDA is disabled; skipping remainder of test." << endl;
    return;
#endif

    vec = Vec_t(1024);
    EXPECT_EQ(1024, vec.size());
    {
        Vec_t other(128);
        int*  orig_vec   = vec.device_ref().data();
        int*  orig_other = other.device_ref().data();
        swap(vec, other);
        EXPECT_EQ(1024, other.size());
        EXPECT_EQ(orig_other, vec.device_ref().data());
        EXPECT_EQ(orig_vec, other.device_ref().data());
    }
    EXPECT_EQ(128, vec.size());

    std::vector<int> data(vec.size());
    data.front() = 1;
    data.back()  = 1234567;

    vec.copy_to_device(celeritas::make_span(data));

    std::vector<int> newdata(vec.size());
    vec.copy_to_host(celeritas::make_span(newdata));
    EXPECT_EQ(1, newdata.front());
    EXPECT_EQ(1234567, newdata.back());

    // Test move construction/assignment
    {
        int*  orig_vec = vec.device_ref().data();
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
TEST_F(DeviceVectorTest, should_not_compile)
{
    DeviceVector<int>    dv(123);
    celeritas::Span<int> s = make_span(dv);
    EXPECT_EQ(123, s.size());

    const auto&                dv_cref = dv;
    celeritas::Span<const int> s2      = make_span(dv_cref);
    EXPECT_EQ(123, s2.size());
}
#endif
