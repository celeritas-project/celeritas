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
    std::vector<uint_t> flattened(11 * 8);
    ASSERT_EQ(flattened.size() * sizeof(uint_t),
              state_ref.size() * sizeof(RanluxppRngState));
    ASSERT_TRUE(std::is_standard_layout_v<RanluxppRngState>);
    ASSERT_EQ(0, offsetof(RanluxppRngState, state));
    ASSERT_EQ(9 * sizeof(RanluxppUInt), offsetof(RanluxppRngState, carry));
    ASSERT_EQ(10 * sizeof(RanluxppUInt), offsetof(RanluxppRngState, position));
    std::copy_n(
        &state_ref.begin()->state[0], flattened.size(), flattened.begin());
    static uint_t const expected_flattened[] = {
        17264205992985481816u,
        664333105563790924u,
        16665412301469266058u,
        3738375993132675275u,
        9714612135453879004u,
        1249755969546447689u,
        8687098122318193493u,
        10859003652624439522u,
        12388405097016529058u,
        1u,
        0u,
        16893635297511449450u,
        4542857486025708045u,
        8045516751372256144u,
        69625187991350341u,
        13501338818076013446u,
        9168488109848218617u,
        1321135559702410711u,
        5093535234471766604u,
        17063017779030278154u,
        1u,
        0u,
        16891145945626919105u,
        17810600685073912687u,
        5639231135592078515u,
        11579219523574949883u,
        16037362265726553367u,
        8432746099412442226u,
        2734228056223852521u,
        17519657097618835891u,
        12121188971786030693u,
        1u,
        0u,
        5000515644769438963u,
        1707214235144974427u,
        2622710761319147939u,
        12234527210822703063u,
        14661054552867259242u,
        10182819665974049105u,
        9736117841522788944u,
        1897415921214275367u,
        10336963720555961685u,
        0u,
        0u,
        4577546626222575745u,
        5332301740947516285u,
        15235279709915406359u,
        6045928354703885979u,
        12788910305413648278u,
        14987711653573464663u,
        2233320969799622527u,
        7505859210888995731u,
        17803156138523120113u,
        1u,
        0u,
        2703848857513886723u,
        6610793152719819544u,
        1677218393582293445u,
        7723588980562403685u,
        7613324377515250742u,
        4550134052925769242u,
        16417108088268416939u,
        6448944518036505345u,
        5050967396761919602u,
        0u,
        0u,
        7172018085051894405u,
        11850851365410869532u,
        1444538108807873810u,
        8960192987265778639u,
        6182102619984829137u,
        16472348632868506945u,
        9931162439855168634u,
        16965344500357381795u,
        1709368374091281522u,
        0u,
        0u,
        4673857085959929601u,
        10283533421962832137u,
        13870535157958825254u,
        10683670844584767371u,
        1381769202635983486u,
        1291540369305179749u,
        4203327276448935978u,
        11998809430000103049u,
        8610076201675217176u,
        1u,
        0u,
    };
    EXPECT_VEC_EQ(expected_flattened, flattened);
}

TEST_F(RanluxppRngEngineTest, host_stream)
{
    // Construct and initialize on "another thread"
    HostStore states(params_->host_ref(), StreamId{1}, 8);

    Span<RanluxppRngState> state_ref
        = states.ref().state[AllItems<RanluxppRngState>{}];
    std::vector<uint_t> flattened(11 * 8);
    std::copy_n(
        &state_ref.begin()->state[0], flattened.size(), flattened.begin());
    static uint_t const expected_flattened[] = {3720570915022148044u,
                                                17699722620430889241u,
                                                11647176959067266033u,
                                                2532577725433413897u,
                                                8944179839461500966u,
                                                10378837391551182728u,
                                                6410025300044534017u,
                                                11641966212450879641u,
                                                14083262405524599887u,
                                                1u,
                                                0u,
                                                455135512892399651u,
                                                17023474611191147117u,
                                                17373258919774478888u,
                                                17576445223244636528u,
                                                15847913825802281056u,
                                                9842258007384848051u,
                                                8993809214237335680u,
                                                10811111478284474722u,
                                                12246841492056226176u,
                                                0u,
                                                0u,
                                                10463657291610495557u,
                                                14531615596319176360u,
                                                4467215711660096154u,
                                                2132154965727001485u,
                                                10816840553003945895u,
                                                11350228709984678334u,
                                                5648319575209995162u,
                                                16862673555938731809u,
                                                11001688364132894656u,
                                                0u,
                                                0u,
                                                3223180854317729964u,
                                                13142865748654267678u,
                                                5233879176515363956u,
                                                14382018580741786076u,
                                                17015426227645395457u,
                                                10235314236713270159u,
                                                1081899657831287459u,
                                                2501347865116569594u,
                                                7402781096869358379u,
                                                0u,
                                                0u,
                                                15229740417035923378u,
                                                9944837482406040861u,
                                                16113942309648519591u,
                                                16305632808613254930u,
                                                10596445834200706676u,
                                                2386607565746041196u,
                                                2298231160130545132u,
                                                817702779694355545u,
                                                3273163631843193162u,
                                                0u,
                                                0u,
                                                11843519348749704946u,
                                                4029433470101260580u,
                                                14177835058926040212u,
                                                7195641552531231367u,
                                                14800969996344824706u,
                                                10727362426596974308u,
                                                10257219530343644822u,
                                                2120455867274771895u,
                                                3292628653710146562u,
                                                0u,
                                                0u,
                                                6787192902998389141u,
                                                7630152796153712738u,
                                                18083355582017085913u,
                                                11553540531576224747u,
                                                14092302672721972057u,
                                                8104229853217942650u,
                                                14720252546972946527u,
                                                1421618926864755509u,
                                                11836820413943009986u,
                                                0u,
                                                0u,
                                                2724599869306841861u,
                                                11966608267318555957u,
                                                3857756632018748671u,
                                                17914390814995526025u,
                                                4965598897176912278u,
                                                1708503242055098632u,
                                                7182205733410213079u,
                                                13580001877620804856u,
                                                2473755285284327500u,
                                                1u,
                                                0u};
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
