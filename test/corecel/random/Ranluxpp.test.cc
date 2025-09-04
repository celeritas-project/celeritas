//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/Ranluxpp.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/engine/Ranluxpp.hh"

#include <memory>

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
    /*
    using HostStore = CollectionStateStore<XorwowRngStateData, MemSpace::host>;
      using DeviceStore
          = CollectionStateStore<XorwowRngStateData, MemSpace::device>;
      using uint_t = XorwowUInt;
  */

    void SetUp() override
    {
        engine_ = std::make_shared<RanluxppDouble>(12345);
    }

    std::shared_ptr<RanluxppDouble> engine_;
};

TEST_F(RanluxppRngEngineTest, host_stream)
{
    std::vector<RanluxppUInt> samples;
    unsigned int num_samples = 8 * 6;
    for (auto i : celeritas::range(num_samples))
    {
        // std::cout << (*engine_)() << std::endl;
        // std::cout << generate_canonical(*engine_);
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

    RngTally tally;

    for (unsigned int i = 0; i < num_seeds; ++i)
    {
        // RanluxppDouble rng(params_->host_ref(), states.ref(),
        // TrackSlotId{i});
        for (unsigned int j = 0; j < num_samples; ++j)
        {
            tally(generate_canonical(*engine_));
        }
    }
    tally.check(num_samples * num_seeds, 1e-3);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
