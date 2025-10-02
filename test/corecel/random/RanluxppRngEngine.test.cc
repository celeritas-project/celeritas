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
    std::vector<uint_t> flattened(10 * 8);
    //(9 * sizeof(RanluxppUInt) + sizeof(unsigned int) + sizeof(int)) * 8);
    ASSERT_EQ(flattened.size() * sizeof(uint_t),
              state_ref.size() * sizeof(RanluxppRngState));
    ASSERT_TRUE(std::is_standard_layout_v<RanluxppRngState>);
    ASSERT_EQ(0, offsetof(RanluxppRngState, state));
    ASSERT_EQ(9 * sizeof(RanluxppUInt), offsetof(RanluxppRngState, carry));
    ASSERT_EQ(9 * sizeof(RanluxppUInt) + sizeof(unsigned int),
              offsetof(RanluxppRngState, position));
    std::copy_n(
        &state_ref.begin()->state[0], flattened.size(), flattened.begin());
    static uint_t const expected_flattened[] = {
        15633181524457117234u, 4048639090307458978u,
        7105913838505417480u,  17602867426280377497u,
        4664185668681883142u,  6801277489608455105u,
        527625378205703086u,   17134196838734872791u,
        5190317336058948500u,  0u,
        3126769704021474677u,  4569240147138733689u,
        13416000272778254460u, 1678010933852105869u,
        15995300448495413899u, 4973951214094681423u,
        14597339503865416801u, 16979346228683075508u,
        12598294308020826281u, 1u,
        9670598024290087806u,  18142382567057873350u,
        13620617593237246881u, 388893018918904112u,
        17079164819033104950u, 16729577827452795550u,
        6771447431402201472u,  8258700677970410321u,
        5936511748377753757u,  1u,
        3874997128947836054u,  8946121475059206304u,
        7224173179413858048u,  4167801349632768010u,
        17750132599320352211u, 11675987299101180990u,
        8173966644759076934u,  11123864335656493313u,
        1941774767182728768u,  1u,
        18383135287682755016u, 5749730882258172408u,
        12173205110483971794u, 4916339591911639153u,
        9510247599756233121u,  15462239984897097784u,
        385069123633977013u,   3412317276703403773u,
        6441945998758586420u,  1u,
        4681748443834643566u,  14704046475199184816u,
        16181193039526776603u, 16611994615453652711u,
        17432571814395710366u, 11480362963330766380u,
        9386370308050554461u,  1984199551442989341u,
        778231190613796011u,   0u,
        3523943565933605286u,  4523090582987778514u,
        7941981923400239324u,  13665411456876568827u,
        3299179412316646176u,  15192696855409109957u,
        8268131408667425089u,  30495338366101166u,
        2853568207021759683u,  0u,
        6589865973559850671u,  15842140447686106450u,
        15956151670782018126u, 17149633609026867699u,
        14898172453466931119u, 8800995199942308013u,
        18249837537690977207u, 3666427774943648075u,
        11763816537044994302u, 0u,
    };
    EXPECT_VEC_EQ(expected_flattened, flattened);
}

TEST_F(RanluxppRngEngineTest, host_stream)
{
    // Construct and initialize on "another thread"
    HostStore states(params_->host_ref(), StreamId{1}, 8);

    Span<RanluxppRngState> state_ref
        = states.ref().state[AllItems<RanluxppRngState>{}];
    std::vector<uint_t> flattened(10 * 8);
    std::copy_n(
        &state_ref.begin()->state[0], flattened.size(), flattened.begin());
    static uint_t const expected_flattened[] = {
        8756617977204430216u,  1844435084667567841u,
        12184035111576480337u, 1044411840907476159u,
        14494229379375184950u, 14906908948396447825u,
        9448299591834748846u,  4956629645495003697u,
        10086829679668943178u, 1u,
        15338717268108967535u, 10125999190981379524u,
        3850105217048512160u,  15500902891759434315u,
        10537018118650080852u, 7975576375037910358u,
        9309308024462805993u,  14067391433493938752u,
        4791482830834302640u,  0u,
        2070775829441348539u,  5809661579350645258u,
        9642032612830954436u,  3126670411258534544u,
        3485075524619314694u,  11136405222049569191u,
        8261887711141798241u,  15094025379845052446u,
        5043393744977975277u,  0u,
        12190121145692046554u, 15481523387916343363u,
        11676846301396059785u, 4279881412522138291u,
        7313479942099795406u,  7091804895415046215u,
        11807977261895291255u, 15666144917392272002u,
        18252684987377536835u, 1u,
        2954461084900225268u,  4466569571487536110u,
        4216552631184363790u,  7318933459704697269u,
        3167448009874483214u,  15783423010903776513u,
        14820405017075508666u, 16355122608290721022u,
        14371433570524119075u, 1u,
        2120186871297035658u,  15405064469561498524u,
        8029729456143067870u,  4375383987952354896u,
        15612412676209896276u, 6553589190780921574u,
        16296173335856849938u, 1302523575230243362u,
        13237214426936422398u, 1u,
        609671935392508045u,   8094343309288548109u,
        2188474879160328502u,  15279964433928005879u,
        8004664956646903248u,  5591791279272682066u,
        14477031318525797565u, 15762673470300806758u,
        6163696117174613141u,  0u,
        11495982577716233659u, 4558786673294232032u,
        11039782028552024269u, 9084550032802791415u,
        8314211526643668155u,  8836946889018723977u,
        17112769085434130164u, 1764664412844316427u,
        4547756696472658029u,  0u,
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
    rng = 12345;
    skip_rng = 12345;

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

    EXPECT_VEC_EQ(ref_host_state.data()->state, host_state.data()->state);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
