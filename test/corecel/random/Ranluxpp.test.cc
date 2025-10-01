//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/Ranluxpp.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/engine/Ranluxpp.hh"

#include <memory>

#include "corecel/data/CollectionStateStore.hh"
#include "corecel/random/data/RanluxppRngData.hh"
#include "corecel/random/data/RanluxppTypes.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"

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
        = CollectionStateStore<RanluxppRngStateData, MemSpace::host>;
    using DeviceStore
        = CollectionStateStore<RanluxppRngStateData, MemSpace::device>;
    using uint_t = RanluxppUInt;

    void SetUp() override
    {
        params_ = std::make_shared<RanluxppRngParams>(12345);
    }

    std::shared_ptr<RanluxppDouble> params_;
};

TEST_F(RanluxppRngEngineTest, host)
{
    // Construct and initialize
    HostStore states(params_->host_ref(), StreamId{0}, 8);

    Span<RanluxppRngState> state_ref
        = states.ref().state[AllItems<RanluxppRngState>{}];

    // Check that initial states are reproducibly random by reading the data as
    // a raw array of uints
    std::vector<uint_t> flattened(11 * 8);
    ASSERT_EQ(flattened.size() * sizeof(uint_t),
              state_ref.size() * sizeof(RanluxppUInt));
    ASSERT_TRUE(std::is_standard_layout_v<RanluxppRngState>);
    ASSERT_EQ(0, offsetof(RanluxppRngState, state));
    ASSERT_EQ(9 * sizeof(RanluxppUInt), offsetof(RanluxppRngState, carry));
    ASSERT_EQ(10 * sizeof(RanluxppUInt), offsetof(RanluxppRngState, position));
    std::copy_n(
        &state_ref.begin()->state[0], flattened.size(), flattened.begin());

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

TEST_F(RanluxppRngEngineTest, host_stream)
{
    std::vector<RanluxppUInt> samples;
    unsigned int num_samples = 8 * 6;
    for ([[maybe_unused]] auto i : celeritas::range(num_samples))
    {
        samples.push_back((*engine_)());
    }

    std::vector<RanluxppUInt> vals
        = {4392207277351116676u,  8851106002112092841u,  6687165001508326944u,
           14649595306718238583u, 6007792660552975608u,  1682971753332617272u,
           9048218912539407546u,  14898349330881593755u, 2154255722023496316u,
           14876603361676467033u, 6540826952230197881u,  10051025297400354359u,
           2368527859797673251u,  18243114625125359466u, 15133936008620691633u,
           13915398152001925263u, 12576052509051454656u, 9743048374735624128u,
           17385910253228683137u, 8470791970347852530u,  3290475052660681413u,
           9144435820346813485u,  3367529474631342494u,  10254938359554205951u,
           18348389005005740822u, 7763172532087333568u,  6164167268954246774u,
           11461325386993242143u, 4664236411739052365u,  4801490950126386455u,
           1313471171266290550u,  12395239732905840443u, 9909143980845758819u,
           8033188693182466529u,  5043159544552303040u,  17933364018111042937u,
           10535341925352800920u, 5618558921847149566u,  2520862122056305670u,
           896088653020459168u,   15492382802122875140u, 7850601124589362965u,
           11597299757137510279u, 10888380905635981994u, 855813428685582687u,
           10989995351054180082u, 17889534949146630696u, 845037862102328855u};
    EXPECT_VEC_EQ(vals, samples);
}

TEST_F(RanluxppRngEngineTest, moments)
{
    unsigned int num_samples = 1 << 12;
    unsigned int num_seeds = 1 << 8;

    HostStore states(params_->host_ref(), StreamId{0}, num_seeds);
    RngTally tally;

    for (unsigned int i = 0; i < num_seeds; ++i)
    {
        RanluxppDouble rng(params_->host_ref(), states.ref(), TrackSlotId{i});
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
    XorwowRngEngine rng(params_->host_ref(), states.ref(), TrackSlotId{0});
    XorwowRngEngine skip_rng(params_->host_ref(), states.ref(), TrackSlotId{1});

    XorwowRngInitializer init;
    init.seed = {12345};
    init.subsequence = 0;
    init.offset = 0;
    rng = init;

    for (ull_int offset = 0; offset <= (1 << 16); offset++)
    {
        // Initialize and skip ahead \c offset steps, equivalent to calling
        // next() \c offset times
        init.offset = offset;
        skip_rng = init;
        ASSERT_EQ(rng(), skip_rng());
    }
    for (ull_int count : {4, 21, 170, 65535})
    {
        // Skip ahead without initializing
        skip_rng.discard(count);
        for (ull_int i = 0; i < count; ++i)
        {
            rng();
        }
        EXPECT_EQ(rng(), skip_rng());
    }
    {
        init.subsequence = (1 << 19);
        init.offset = 0;
        rng = init;

        init.subsequence += 1;
        init.offset = 1023;
        skip_rng = init;

        // Skip 2**67 times to get to the next subsequence
        for (size_type i = 0; i < 8; ++i)
        {
            rng.discard(numeric_limits<unsigned long long>::max());
            rng.discard(1);
        }
        // Skip to the right offset
        rng.discard(init.offset);

        EXPECT_EQ(rng(), skip_rng());
    }
}

TEST_F(XorwowRngEngineTest, TEST_IF_CELER_DEVICE(device))
{
    // Create and initialize states
    DeviceStore rng_store(params_->host_ref(), StreamId{0}, 1024);
    // Copy to host and check
    StateCollection<XorwowState, Ownership::value, MemSpace::host> host_state;
    host_state = rng_store.ref().state;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
