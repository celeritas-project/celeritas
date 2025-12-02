//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/Ranluxpp.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/engine/RanluxppRngEngine.hh"

#include <memory>

#include "corecel/data/CollectionStateStore.hh"
#include "corecel/random/data/RanluxppRngData.hh"
#include "corecel/random/data/RanluxppTypes.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "corecel/random/params/RanluxppRngParams.hh"

#include "RngTally.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class RanluxppRngEngineTest : public Test
{
  protected:
    using HostStore
        = CollectionStateStore<celeritas::RanluxppRngStateData, MemSpace::host>;
    using DeviceStore
        = CollectionStateStore<celeritas::RanluxppRngStateData, MemSpace::device>;
    using uint_t = RanluxppUInt;

    void SetUp() override
    {
        params_ = std::make_shared<RanluxppRngParams>(12345);
    }

    std::shared_ptr<RanluxppRngParams> params_;
};

TEST_F(RanluxppRngEngineTest, host)
{
    // Construct and initialize
    HostStore states(params_->host_ref(), StreamId{0}, 8);

    Span<RanluxppRngState> state_ref
        = states.ref().state[AllItems<RanluxppRngState>{}];

    // Check that initial states are reproducibly random by reading the data as
    // a raw array of uints
    std::vector<uint_t> flattened;
    for (RanluxppRngState const& s : state_ref)
    {
        flattened.insert(
            flattened.end(), s.value.number.begin(), s.value.number.end());
        flattened.push_back(s.value.carry);
        flattened.push_back(s.position);
    }
    static uint_t const expected_flattened[] = {
        9744429461633961477ull,
        11964953901972274469ull,
        4478520638286775329ull,
        16026674858637250013ull,
        5868161582470726065ull,
        18083443687057146648ull,
        2333823551862610090ull,
        4810170359328042893ull,
        9219793804842481641ull,
        1ull,
        0ull,
        5996244813776094669ull,
        15514079292022756606ull,
        13070503915723668046ull,
        16054146445297325220ull,
        77364431278731566ull,
        12455179013329781556ull,
        4960946289522070044ull,
        15803305577134101177ull,
        15815960628694634594ull,
        1ull,
        0ull,
        5292029055920796929ull,
        354307190512697262ull,
        826025980037141022ull,
        9682419061478460462ull,
        3151222553611116004ull,
        16677841673370297929ull,
        11192170787848258518ull,
        13614278094310384612ull,
        17186632708803673196ull,
        1ull,
        0ull,
        10389849264529850277ull,
        12003237817088076700ull,
        5710500540437556604ull,
        14590571546846764795ull,
        8060510615989267314ull,
        2397923598432963768ull,
        1048438652322521925ull,
        10290687144196975613ull,
        16715650114672424133ull,
        1ull,
        0ull,
        13887981414009315101ull,
        12328490092871263700ull,
        3155137450628681419ull,
        15811951024040657914ull,
        6768987711711634516ull,
        7755474955751860888ull,
        2606193410039397811ull,
        9227034287710565885ull,
        8679252532287362054ull,
        1ull,
        0ull,
        3855945920205696888ull,
        13358958311475123590ull,
        6710764668229434537ull,
        4262848890307254740ull,
        1279799399919431628ull,
        536225618428374571ull,
        8508332201360500246ull,
        3462380046783735749ull,
        1800074275881723809ull,
        0ull,
        0ull,
        5344734045182833169ull,
        13065802197196896331ull,
        11834702750481817224ull,
        693680568123867935ull,
        13722072697020217490ull,
        13469907416056267777ull,
        10056982090538872080ull,
        6076277212808993782ull,
        10399244753072501130ull,
        1ull,
        0ull,
        24947320461023194ull,
        8960620023439338087ull,
        5748963559649237544ull,
        9261125454644926276ull,
        669497721652243354ull,
        10999822858694997285ull,
        9643800529102577235ull,
        12711219403559625423ull,
        6130403261299341180ull,
        0ull,
        0ull,
    };
    EXPECT_VEC_EQ(expected_flattened, flattened);
}

TEST_F(RanluxppRngEngineTest, moments)
{
    unsigned int num_samples = 1 << 13;
    unsigned int num_seeds = 1 << 8;

    HostStore states(params_->host_ref(), StreamId{0}, num_seeds);
    RngTally tally;

    for (unsigned int i = 0; i < num_seeds; ++i)
    {
        RanluxppRngEngine rng(
            params_->host_ref(), states.ref(), TrackSlotId{i});
        for (unsigned int j = 0; j < num_samples; ++j)
        {
            tally(generate_canonical(rng));
        }
    }
    tally.check(num_samples * num_seeds, 1e-3);
}

TEST_F(RanluxppRngEngineTest, jump)
{
    unsigned int size = 2;

    HostStore states(params_->host_ref(), StreamId{0}, size);
    RanluxppRngEngine rng(params_->host_ref(), states.ref(), TrackSlotId{0});
    RanluxppRngEngine skip_rng(
        params_->host_ref(), states.ref(), TrackSlotId{1});
    rng = RanluxppInitializer{12345, 0, 0};
    skip_rng = RanluxppInitializer{12345, 0, 0};

    // Compare first 5 random numbers
    for ([[maybe_unused]] int i : celeritas::range(5))
    {
        EXPECT_EQ(rng(), skip_rng());
    }

    // Draw 10 additional random numbers from rng
    for ([[maybe_unused]] int i : celeritas::range(10))
    {
        rng();
    }

    // Discard 10 numbers of skip_rng
    skip_rng.discard(10);

    // Draw the next 20 random numbers and compare
    for ([[maybe_unused]] int i : celeritas::range(20))
    {
        EXPECT_EQ(rng(), skip_rng());
    }
}

TEST_F(RanluxppRngEngineTest, TEST_IF_CELER_DEVICE(device))
{
    celeritas::device().create_streams(1);

    // Create and initialize states
    DeviceStore rng_store(params_->host_ref(), StreamId{0}, 1024);

    // Copy to host
    StateCollection<RanluxppRngState, Ownership::value, MemSpace::host> host_state;
    host_state = rng_store.ref().state;

    // Create and initialize states on host
    HostStore ref_rng_store(params_->host_ref(), StreamId{0}, 1024);
    StateCollection<RanluxppRngState, Ownership::value, MemSpace::host>
        ref_host_state;
    ref_host_state = ref_rng_store.ref().state;

    EXPECT_VEC_EQ(ref_host_state.data()->value.number,
                  host_state.data()->value.number);
    EXPECT_EQ(ref_host_state.data()->value.carry,
              host_state.data()->value.carry);
    EXPECT_EQ(ref_host_state.data()->position, host_state.data()->position);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
