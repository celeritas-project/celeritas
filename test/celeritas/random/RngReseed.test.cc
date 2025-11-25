//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngReseed.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/random/RngReseed.hh"

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/random/engine/RngEngine.hh"
#include "corecel/random/params/RngParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class RngReseedTest : public Test
{
  public:
    using RngHostStore = CollectionStateStore<RngStateData, MemSpace::host>;

    void SetUp() override { params = std::make_shared<RngParams>(12345); }

    std::shared_ptr<RngParams> params;
};

TEST_F(RngReseedTest, reseed)
{
    // Create and initialize states
    size_type size = 1024;
    RngHostStore states(params->host_ref(), StreamId{0}, size);

    size_type id = 8;
    reseed_rng(
        params->host_ref(), states.ref(), StreamId{0}, UniqueEventId{id});

    RngEngine::Initializer_t init;
    init.seed = params->host_ref().seed;
    init.offset = 0;

    for (size_type i = 1; i < states.size(); ++i)
    {
        // Check that the reseeded RNGs were skipped ahead the correct number
        // of subsequences
        RngEngine skip_rng(params->host_ref(), states.ref(), TrackSlotId{0});
        init.subsequence = id * states.size() + i;
        skip_rng = init;

        RngEngine rng(params->host_ref(), states.ref(), TrackSlotId{i});
        ASSERT_EQ(skip_rng(), rng());
    }

    reseed_rng(
        params->host_ref(), states.ref(), StreamId{0}, UniqueEventId{id});
    std::vector<unsigned int> values;
    for (auto i : range(states.size()).step(128u))
    {
        RngEngine rng(params->host_ref(), states.ref(), TrackSlotId{i});
        values.push_back(rng());
    }
#if CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_CURAND
    static unsigned int const expected_values[] = {65145249u,
                                                   4154590960u,
                                                   2216085262u,
                                                   241608182u,
                                                   2278993841u,
                                                   1643630301u,
                                                   2759037535u,
                                                   3550652068u};
#elif CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW
    static unsigned int const expected_values[] = {3522223652u,
                                                   296995412u,
                                                   1414776235u,
                                                   1609101469u,
                                                   363980503u,
                                                   2861073075u,
                                                   1771581540u,
                                                   3600889717u};
#elif CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_RANLUXPP
    static unsigned int const expected_values[] = {1506987910u,
                                                   289122019u,
                                                   3660744945u,
                                                   46319873u,
                                                   2718473539u,
                                                   1739478500u,
                                                   3043999655u,
                                                   2441727977u};
#endif
    EXPECT_VEC_EQ(values, expected_values);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
