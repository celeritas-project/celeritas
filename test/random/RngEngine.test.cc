//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngEngine.test.cc
//---------------------------------------------------------------------------//
#include "random/RngParams.hh"

#include "celeritas_test.hh"
#include "RngEngine.test.hh"
#include "base/CollectionStateStore.hh"

using celeritas::CollectionStateStore;
using celeritas::RngParams;
using celeritas::RngStateData;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class RngEngineTest : public celeritas::Test
{
  public:
    using RngDeviceStore = CollectionStateStore<RngStateData, MemSpace::device>;

    void SetUp() override { params = std::make_shared<RngParams>(12345); }

    std::shared_ptr<RngParams> params;
};

TEST_F(RngEngineTest, TEST_IF_CELERITAS_CUDA(device))
{
    // Create and initialize states
    RngDeviceStore rng_store(*params, 1024);

    // Generate on device
    std::vector<unsigned int> values = re_test_native(rng_store.ref());

    // Print a subset of the values
    std::vector<unsigned int> test_values;
    for (auto i : celeritas::range(rng_store.size()).step(127u))
    {
        test_values.push_back(values[i]);
    }

    // PRINT_EXPECTED(test_values);
    static const unsigned int expected_test_values[] = {165860337u,
                                                        3006138920u,
                                                        2161337536u,
                                                        390101068u,
                                                        2347834113u,
                                                        100129048u,
                                                        4122784086u,
                                                        473544901u,
                                                        2822849608u};
    EXPECT_VEC_EQ(test_values, expected_test_values);
}

//---------------------------------------------------------------------------//
// FLOAT TEST
//---------------------------------------------------------------------------//

template<typename T>
class RngEngineFloatTest : public RngEngineTest
{
};

void check_expected_float_samples(const std::vector<float>& v)
{
    ASSERT_LE(2, v.size());
    EXPECT_FLOAT_EQ(0.038617369, v[0]);
    EXPECT_FLOAT_EQ(0.411269426, v[1]);
}

void check_expected_float_samples(const std::vector<double>& v)
{
    ASSERT_LE(2, v.size());
    EXPECT_DOUBLE_EQ(0.283318433931184, v[0]);
    EXPECT_DOUBLE_EQ(0.653335242131673, v[1]);
}

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RngEngineFloatTest, FloatTypes, );

#if CELERITAS_USE_CUDA
TYPED_TEST(RngEngineFloatTest, device)
#else
TYPED_TEST(RngEngineFloatTest, DISABLED_device)
#endif
{
    using RngDeviceStore = typename TestFixture::RngDeviceStore;
    using real_type      = TypeParam;

    // Create and initialize states
    RngDeviceStore rng_store(*this->params, 100);

    // Generate on device
    auto values = re_test_canonical<real_type>(rng_store.ref());

    // Test result
    EXPECT_EQ(rng_store.size(), values.size());
    for (real_type sample : values)
    {
        EXPECT_GE(sample, real_type(0));
        EXPECT_LT(sample, real_type(1));
    }

    check_expected_float_samples(values);
}
