//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/engine/CachedRngEngine.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/engine/CachedRngEngine.hh"

#include <random>

#include "corecel/data/CollectionStateStore.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "corecel/random/engine/XorwowRngEngine.hh"
#include "corecel/random/params/XorwowRngParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

template<class T, class E>
std::vector<T> sample(size_type count, E& engine)
{
    UniformRealDistribution<T> sample_real;
    std::vector<T> result(count);
    for (auto& r : result)
    {
        r = sample_real(engine);
    }
    return result;
}

//---------------------------------------------------------------------------//

class CachedRngEngineTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(CachedRngEngineTest, mt)
{
    auto expected = [] {
        std::mt19937 rng;
        return sample<double>(8, rng);
    }();

    auto cached = [] {
        // 32-bit rng, 64-bit value
        std::mt19937 rng;

        auto result = cache_rng_values<double, 8>(rng);
        EXPECT_EQ(2 * 8, result.size());
        EXPECT_EQ(result.remaining(), result.size());
        return result;
    }();

    auto actual = sample<double>(8, cached);
    EXPECT_EQ(0, cached.remaining());
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(cached(), DebugError);
    }

    EXPECT_VEC_EQ(expected, actual);
}

TEST_F(CachedRngEngineTest, xorwow)
{
    using HostStore = CollectionStateStore<XorwowRngStateData, MemSpace::host>;

    // Construct params with seed
    XorwowRngParams params{12345};
    // Construct a single RNG
    HostStore states(params.host_ref(), StreamId{0}, 1);

    auto expected = [&] {
        XorwowRngEngine rng(params.host_ref(), states.ref(), TrackSlotId{0});
        return sample<double>(4, rng);
    }();

    // Reinitialize the RNG
    initialize_xorwow(states.ref().state[AllItems<XorwowState>{}],
                      params.host_ref().seed,
                      StreamId{0});

    auto cached = [&] {
        XorwowRngEngine rng(params.host_ref(), states.ref(), TrackSlotId{0});
        auto result = cache_rng_values<double, 4>(rng);
        EXPECT_EQ(2 * 4, result.size());
        EXPECT_EQ(result.remaining(), result.size());
        return result;
    }();

    auto actual = sample<double>(4, cached);
    EXPECT_EQ(0, cached.remaining());

    EXPECT_VEC_EQ(expected, actual);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
