//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/XorwowRngEngine.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/random/XorwowRngEngine.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <string>
#include <type_traits>

#include "corecel/data/CollectionStateStore.hh"
#include "corecel/io/detail/ReprImpl.hh"
#include "celeritas/random/XorwowRngParams.hh"
#include "celeritas/random/detail/GenerateCanonical32.hh"

#include "HexRepr.hh"
#include "RngTally.hh"
#include "SequenceEngine.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
using ::celeritas::test::SequenceEngine;

TEST(GenerateCanonical32, flt)
{
    using std::nextafter;
    // Test numbers at beginning, midpoint, and end
    SequenceEngine rng({0x00000000u,
                        0x00000001u,
                        0x7fffffffu,
                        0x80000000u,
                        0x80000001u,
                        0xfffffffeu,
                        0xffffffffu});

    // NOTE: hexadecimal floating point literals are a feature of C++17, so we
    // have to work around with "stof"/"stod"
    GenerateCanonical32<float> generate_canonical;
    EXPECT_FLOAT_EQ(0.0f, generate_canonical(rng));
    EXPECT_FLOAT_EQ(std::stof("0x0.00000001p0"), generate_canonical(rng));
    EXPECT_FLOAT_EQ(nextafter(0.5f, 0.0f), generate_canonical(rng));
    EXPECT_FLOAT_EQ(0.5f, generate_canonical(rng));
    EXPECT_FLOAT_EQ(nextafter(0.5f, 1.0f), generate_canonical(rng));
    EXPECT_FLOAT_EQ(std::stof("0x0.fffffffep0"), generate_canonical(rng));
    EXPECT_FLOAT_EQ(nextafter(1.0f, 0.0f), generate_canonical(rng));
}

TEST(GenerateCanonical32, dbl)
{
    using ::celeritas::test::hex_repr;
    // Upper/[xor|lower] bits
    // clang-format off
    SequenceEngine rng({0x00000000u, 0x00000000u, // 0
                        0x00000000u, 0x00000001u,
                        0x00000001u, 0x00000000u,
                        0x80000000u, 0x00000000u, // 0.5
                        0xffffffffu, 0xffffffffu,
                        0xffffffffu, 0x001ffffeu,
                        0xffffffffu, 0x001fffffu,});
    // clang-format on

    GenerateCanonical32<double> generate_canonical;

    auto actual = generate_canonical(rng);
    ASSERT_TRUE((std::is_same<double, decltype(actual)>::value));
    EXPECT_EQ(0.0, actual) << hex_repr(actual);

    actual = generate_canonical(rng);
    EXPECT_EQ(std::stod("0x0.00000000000008p0"), actual) << hex_repr(actual);

    actual = generate_canonical(rng);
    EXPECT_EQ(std::stod("0x1p-32"), actual) << hex_repr(actual);

    actual = generate_canonical(rng);
    EXPECT_EQ(0.5, actual) << hex_repr(actual);

    actual = generate_canonical(rng);
    EXPECT_EQ(std::stod("0x1.fffff001fffffp-1"), actual) << hex_repr(actual);

    actual = generate_canonical(rng);
    EXPECT_EQ(std::stod("0x1.ffffffffffffep-1"), actual) << hex_repr(actual);

    actual = generate_canonical(rng);
    EXPECT_LT(actual, 1.0);
    EXPECT_EQ(std::stod("0x1.fffffffffffffp-1"), actual) << hex_repr(actual);
    EXPECT_EQ(nextafter(1.0, 0.0), actual) << hex_repr(nextafter(1.0, 0.0));
}

TEST(GenerateCanonical32, moments)
{
    int num_samples = 1 << 20;  // ~1m

    std::mt19937 rng;
    GenerateCanonical32<double> generate_canonical;
    ::celeritas::test::RngTally tally;

    for (int i = 0; i < num_samples; ++i)
    {
        tally(generate_canonical(rng));
    }
    tally.check(num_samples, 1e-3);
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail

namespace test
{
//---------------------------------------------------------------------------//
class XorwowRngEngineTest : public Test
{
  protected:
    using HostStore = CollectionStateStore<XorwowRngStateData, MemSpace::host>;
    using DeviceStore
        = CollectionStateStore<XorwowRngStateData, MemSpace::device>;
    using uint_t = XorwowState::uint_t;

    void SetUp() override
    {
        params = std::make_shared<XorwowRngParams>(12345);
    }

    std::shared_ptr<XorwowRngParams> params;
};

TEST_F(XorwowRngEngineTest, host)
{
    // Construct and initialize
    HostStore states(params->host_ref(), 8);

    Span<XorwowState> state_ref = states.ref().state[AllItems<XorwowState>{}];

    // Check that initial states are reproducibly random by reading the data as
    // a raw array of uints
    std::vector<uint_t> flattened(8 * 6);
    ASSERT_EQ(flattened.size() * sizeof(uint_t),
              state_ref.size() * sizeof(XorwowState));
    ASSERT_TRUE(std::is_standard_layout<XorwowState>::value);
    ASSERT_EQ(0, offsetof(XorwowState, xorstate));
    ASSERT_EQ(5 * sizeof(uint_t), offsetof(XorwowState, weylstate));
    std::copy_n(
        &state_ref.begin()->xorstate[0], flattened.size(), flattened.begin());

    static unsigned int const expected_flattened[]
        = {2421091215u, 3647994171u, 2504472727u, 1236778574u, 4083156575u,
           63361926u,   3719645674u, 843467800u,  1265623178u, 295820715u,
           1583721852u, 802677129u,  3794549800u, 1642707272u, 4266580851u,
           2668696688u, 2910059606u, 1707659088u, 3955349927u, 2857721444u,
           2773100230u, 3321656875u, 1176613630u, 909057096u,  4173021154u,
           338389676u,  2806912494u, 1345761716u, 149057928u,  630801564u,
           3118211368u, 3857808320u, 4193588147u, 925742588u,  1585365047u,
           3244057179u, 3428095051u, 118856847u,  945254054u,  2395966273u,
           1370167352u, 1607766504u, 3084411954u, 2675509253u, 2542521715u,
           327503606u,  3527767224u, 154218656u};
    EXPECT_VEC_EQ(expected_flattened, flattened);
}

TEST_F(XorwowRngEngineTest, moments)
{
    unsigned int num_samples = 1 << 12;
    unsigned int num_seeds = 1 << 8;

    HostStore states(params->host_ref(), num_seeds);
    RngTally tally;

    for (unsigned int i = 0; i < num_seeds; ++i)
    {
        XorwowRngEngine rng(states.ref(), TrackSlotId{i});
        for (unsigned int j = 0; j < num_samples; ++j)
        {
            tally(generate_canonical(rng));
        }
    }
    tally.check(num_samples * num_seeds, 1e-3);
}

TEST_F(XorwowRngEngineTest, TEST_IF_CELER_DEVICE(device))
{
    // Create and initialize states
    DeviceStore rng_store(params->host_ref(), 1024);
    // Copy to host and check
    StateCollection<XorwowState, Ownership::value, MemSpace::host> host_state;
    host_state = rng_store.ref().state;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
