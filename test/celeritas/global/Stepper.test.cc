//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/global/Stepper.hh"

#include <cstring>
#include <memory>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/GlobalGeoTestBase.hh"
#include "celeritas/SimpleTestBase.hh"
#include "celeritas/TestEm3Base.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/ActionManager.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "celeritas_cmake_strings.h"
#include "celeritas_test.hh"

using namespace celeritas;

namespace
{
bool string_equal(const char* lhs, const char* rhs)
{
    return std::strcmp(lhs, rhs) == 0;
}
} // namespace

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class DummyAction final : public ExplicitActionInterface, public ConcreteAction
{
  public:
    // Construct with ID and label
    using ConcreteAction::ConcreteAction;

    void execute(CoreHostRef const&) const final { ++num_execute_host_; }
    void execute(CoreDeviceRef const&) const final { ++num_execute_device_; }

    int num_execute_host() const { return num_execute_host_; }
    int num_execute_device() const { return num_execute_device_; }

  private:
    mutable int num_execute_host_{0};
    mutable int num_execute_device_{0};
};

//---------------------------------------------------------------------------//
/*!
 * Construct helper action and set up stepper/primary inputs.
 *
 * This class must be virtual so that it can be used as a mixin to other class
 * definitions.
 */
class StepperTest : virtual public celeritas_test::GlobalTestBase
{
  protected:
    void SetUp() override
    {
        auto& action_mgr = *this->action_mgr();

        static const char desc[] = "count the number of executions";

        dummy_action_ = std::make_shared<DummyAction>(
            action_mgr.next_id(), "dummy-action", desc);
        action_mgr.insert(dummy_action_);
    }

    StepperInput make_stepper_input(size_type tracks, size_type init_scaling)
    {
        CELER_EXPECT(tracks > 0);
        CELER_EXPECT(init_scaling > 1);

        StepperInput result;
        result.params           = this->core();
        result.num_track_slots  = tracks;
        result.num_initializers = init_scaling * tracks;

        CELER_ASSERT(dummy_action_);
        result.post_step_callback = dummy_action_->action_id();
        CELER_ENSURE(result.post_step_callback);
        return result;
    }

    virtual std::vector<Primary> make_primaries(size_type count) const = 0;

    std::shared_ptr<DummyAction> dummy_action_;
};

#if !CELERITAS_USE_GEANT4
#    define TestEm3Test DISABLED_TestEm3Test
#endif
class TestEm3Test : public celeritas_test::TestEm3Base, public StepperTest
{
  public:
    //! Make 10GeV electrons along +x
    std::vector<Primary> make_primaries(size_type count) const final
    {
        Primary p;
        p.particle_id = this->particle()->find("e-");
        CELER_ASSERT(p.particle_id);
        p.energy    = units::MevEnergy{10000};
        p.track_id  = TrackId{0};
        p.position  = {-22, 0, 0};
        p.direction = {1, 0, 0};

        std::vector<Primary> result(count, p);
        for (auto i : range(count))
        {
            result[i].event_id = EventId{i};
        }
        return result;
    }
};

//---------------------------------------------------------------------------//
// TESTEM3
//---------------------------------------------------------------------------//

TEST_F(TestEm3Test, host)
{
    size_type               num_primaries   = 8;
    size_type               inits_per_track = 32;
    size_type               num_tracks      = num_primaries * inits_per_track;
    Stepper<MemSpace::host> step(
        this->make_stepper_input(num_tracks, inits_per_track));

    auto counts = step(this->make_primaries(1));
    EXPECT_EQ(1, counts.active);
    EXPECT_EQ(0, counts.queued);
    EXPECT_EQ(1, counts.alive);

    std::vector<size_type> active = {counts.active};
    std::vector<size_type> queued = {counts.queued};
    while (counts)
    {
        counts = step();
        active.push_back(counts.active);
        queued.push_back(counts.queued);
        ASSERT_LT(active.size(), 1000) << "max iterations exceeded";
    }

    if (string_equal(celeritas_rng, "XORWOW")
        && string_equal(celeritas_clhep_version, "2.4.5.1")
        && string_equal(celeritas_geant4_version, "10.7.3"))
    {
        static const unsigned int expected_active[] = {
            1u,   1u,   2u,   3u,   5u,   7u,   8u,   12u,  16u,  21u,  26u,
            29u,  32u,  35u,  41u,  47u,  55u,  64u,  85u,  102u, 111u, 117u,
            130u, 130u, 140u, 153u, 167u, 185u, 206u, 226u, 239u, 255u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u,
            256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u, 249u, 241u, 228u,
            209u, 208u, 203u, 193u, 190u, 173u, 164u, 158u, 149u, 148u, 151u,
            156u, 155u, 155u, 147u, 135u, 139u, 140u, 129u, 117u, 114u, 113u,
            104u, 98u,  102u, 99u,  88u,  91u,  88u,  83u,  80u,  79u,  75u,
            73u,  76u,  75u,  72u,  66u,  67u,  63u,  63u,  63u,  58u,  56u,
            49u,  39u,  35u,  34u,  31u,  34u,  35u,  32u,  34u,  35u,  33u,
            27u,  29u,  31u,  28u,  30u,  29u,  28u,  21u,  19u,  16u,  15u,
            13u,  13u,  13u,  12u,  9u,   9u,   9u,   9u,   9u,   10u,  12u,
            12u,  13u,  11u,  11u,  12u,  11u,  9u,   10u,  9u,   8u,   7u,
            6u,   3u,   3u,   3u,   3u,   2u,   2u,   1u,   1u};
        EXPECT_VEC_EQ(expected_active, active);

        static const unsigned int expected_queued[]
            = {0u,    1u,    1u,    2u,    2u,    1u,    4u,    4u,    6u,
               7u,    5u,    8u,    5u,    7u,    10u,   9u,    15u,   24u,
               22u,   18u,   21u,   24u,   16u,   19u,   26u,   30u,   30u,
               36u,   34u,   39u,   46u,   52u,   84u,   97u,   124u,  153u,
               196u,  240u,  276u,  312u,  350u,  388u,  426u,  445u,  483u,
               517u,  553u,  583u,  592u,  615u,  625u,  647u,  672u,  691u,
               715u,  745u,  770u,  803u,  829u,  860u,  885u,  916u,  947u,
               970u,  990u,  1024u, 1056u, 1088u, 1106u, 1132u, 1145u, 1161u,
               1185u, 1199u, 1214u, 1241u, 1263u, 1278u, 1276u, 1278u, 1278u,
               1280u, 1274u, 1254u, 1247u, 1247u, 1252u, 1249u, 1234u, 1212u,
               1204u, 1175u, 1174u, 1159u, 1151u, 1138u, 1127u, 1108u, 1094u,
               1079u, 1053u, 1036u, 1030u, 1017u, 1002u, 989u,  972u,  949u,
               928u,  917u,  916u,  908u,  884u,  867u,  846u,  825u,  821u,
               820u,  815u,  805u,  790u,  780u,  763u,  754u,  739u,  730u,
               733u,  721u,  715u,  711u,  697u,  677u,  664u,  652u,  645u,
               643u,  635u,  624u,  617u,  596u,  581u,  565u,  552u,  547u,
               549u,  547u,  551u,  547u,  543u,  541u,  538u,  547u,  536u,
               538u,  535u,  542u,  546u,  539u,  533u,  530u,  510u,  496u,
               480u,  463u,  458u,  449u,  447u,  441u,  446u,  450u,  457u,
               457u,  453u,  443u,  433u,  432u,  409u,  403u,  397u,  395u,
               394u,  386u,  379u,  378u,  384u,  390u,  390u,  387u,  375u,
               376u,  369u,  367u,  362u,  356u,  354u,  341u,  318u,  317u,
               319u,  310u,  295u,  287u,  291u,  289u,  283u,  279u,  284u,
               271u,  262u,  240u,  217u,  196u,  199u,  191u,  183u,  183u,
               177u,  176u,  165u,  141u,  129u,  139u,  138u,  131u,  126u,
               128u,  127u,  116u,  110u,  103u,  100u,  90u,   76u,   68u,
               75u,   60u,   52u,   41u,   28u,   33u,   31u,   22u,   24u,
               25u,   25u,   25u,   19u,   19u,   16u,   15u,   18u,   22u,
               20u,   16u,   25u,   17u,   14u,   16u,   19u,   15u,   17u,
               15u,   15u,   12u,   8u,    13u,   10u,   8u,    16u,   14u,
               8u,    9u,    7u,    8u,    10u,   9u,    11u,   9u,    6u,
               11u,   10u,   9u,    9u,    10u,   9u,    3u,    4u,    3u,
               5u,    1u,    4u,    4u,    2u,    5u,    5u,    2u,    2u,
               3u,    6u,    4u,    6u,    4u,    6u,    3u,    3u,    1u,
               1u,    1u,    1u,    1u,    1u,    0u,    1u,    0u,    1u,
               1u,    1u,    2u,    1u,    3u,    2u,    1u,    2u,    1u,
               0u,    2u,    1u,    1u,    0u,    0u,    0u,    0u,    0u,
               0u,    0u,    0u,    0u,    0u,    0u};
        EXPECT_VEC_EQ(expected_queued, queued);
    }
    else
    {
        cout << "No output saved for combination of RNG=\"" << celeritas_rng
             << "\", CLHEP=\"" << celeritas_clhep_version << "\", Geant4=\""
             << celeritas_geant4_version << "\"\n";
        PRINT_EXPECTED(active);
        PRINT_EXPECTED(queued);
    }

    // Check that callback was called
    EXPECT_EQ(active.size(), dummy_action_->num_execute_host());
    EXPECT_EQ(0, dummy_action_->num_execute_device());
}

TEST_F(TestEm3Test, TEST_IF_CELER_DEVICE(device))
{
    size_type num_primaries   = 8;
    size_type inits_per_track = 1024;
    size_type num_tracks = num_primaries * 800; // low enough to max out the
                                                // active tracks

    Stepper<MemSpace::device> step(
        this->make_stepper_input(num_tracks, inits_per_track));

    auto counts = step(this->make_primaries(num_primaries));
    EXPECT_EQ(num_primaries, counts.active);
    EXPECT_EQ(num_primaries, counts.alive);
    if (std::string(celeritas_rng) == std::string("XORWOW"))
    {
        EXPECT_EQ(0, counts.queued);
    }

    std::vector<size_type> active = {counts.active};
    std::vector<size_type> queued = {counts.queued};
    while (counts)
    {
        counts = step();
        active.push_back(counts.active);
        queued.push_back(counts.queued);
        ASSERT_LT(active.size(), 300) << "max iterations exceeded";
    }

    if (string_equal(celeritas_rng, "XORWOW")
        && string_equal(celeritas_clhep_version, "2.4.5.1")
        && string_equal(celeritas_geant4_version, "10.7.3"))
    {
        static const unsigned int expected_active[]
            = {8u,    8u,    16u,   27u,   35u,   49u,   64u,   74u,   87u,
               99u,   114u,  129u,  164u,  200u,  235u,  285u,  335u,  378u,
               425u,  491u,  544u,  621u,  700u,  799u,  889u,  995u,  1123u,
               1236u, 1359u, 1509u, 1666u, 1832u, 1970u, 2144u, 2333u, 2488u,
               2641u, 2798u, 2980u, 3187u, 3331u, 3485u, 3673u, 3851u, 4022u,
               4201u, 4397u, 4591u, 4848u, 5075u, 5223u, 5449u, 5714u, 5962u,
               6141u, 6297u, 6400u, 6400u, 6400u, 6400u, 6400u, 6400u, 6400u,
               6400u, 6400u, 6400u, 6400u, 6400u, 6400u, 6400u, 6400u, 6400u,
               6400u, 6400u, 6400u, 6400u, 6400u, 6400u, 6400u, 6400u, 6400u,
               6400u, 6400u, 6400u, 6400u, 6400u, 6400u, 6400u, 6400u, 6400u,
               6400u, 6400u, 6400u, 6400u, 6042u, 5819u, 5648u, 5434u, 5225u,
               5049u, 4891u, 4694u, 4566u, 4488u, 4336u, 4246u, 4070u, 3925u,
               3796u, 3640u, 3453u, 3292u, 3128u, 3030u, 2931u, 2822u, 2758u,
               2667u, 2485u, 2418u, 2374u, 2312u, 2188u, 2053u, 1947u, 1848u,
               1685u, 1631u, 1554u, 1484u, 1398u, 1317u, 1246u, 1146u, 1072u,
               999u,  932u,  897u,  804u,  745u,  681u,  615u,  574u,  523u,
               471u,  416u,  372u,  330u,  299u,  297u,  281u,  258u,  226u,
               202u,  183u,  157u,  134u,  119u,  106u,  98u,   92u,   81u,
               76u,   67u,   66u,   62u,   55u,   46u,   41u,   37u,   30u,
               29u,   24u,   18u,   17u,   15u,   17u,   16u,   14u,   14u,
               9u,    6u,    6u,    6u,    5u,    4u,    3u,    3u,    2u,
               2u,    2u,    2u,    2u,    2u,    1u};
        EXPECT_VEC_EQ(expected_active, active);
        static const unsigned int expected_queued[]
            = {0u,    8u,    11u,   8u,    14u,   15u,   12u,   16u,   17u,
               21u,   22u,   42u,   44u,   48u,   58u,   67u,   75u,   76u,
               93u,   95u,   122u,  125u,  157u,  168u,  186u,  223u,  218u,
               241u,  275u,  289u,  313u,  331u,  354u,  414u,  395u,  437u,
               438u,  478u,  540u,  523u,  561u,  605u,  614u,  623u,  627u,
               679u,  726u,  782u,  778u,  806u,  858u,  907u,  941u,  923u,
               961u,  936u,  994u,  1126u, 1289u, 1495u, 1695u, 1886u, 2047u,
               2199u, 2349u, 2529u, 2644u, 2754u, 2888u, 3045u, 3143u, 3217u,
               3336u, 3391u, 3452u, 3407u, 3432u, 3400u, 3366u, 3364u, 3342u,
               3294u, 3201u, 3033u, 2870u, 2674u, 2531u, 2335u, 2129u, 1935u,
               1712u, 1437u, 1089u, 749u,  769u,  726u,  676u,  666u,  666u,
               623u,  581u,  572u,  591u,  545u,  549u,  493u,  483u,  476u,
               441u,  427u,  394u,  368u,  388u,  381u,  333u,  366u,  336u,
               258u,  319u,  282u,  282u,  272u,  246u,  235u,  218u,  171u,
               192u,  175u,  185u,  173u,  155u,  144u,  136u,  120u,  119u,
               98u,   115u,  90u,   86u,   74u,   58u,   67u,   61u,   60u,
               43u,   38u,   33u,   26u,   35u,   32u,   37u,   20u,   18u,
               21u,   15u,   18u,   15u,   12u,   10u,   15u,   6u,    10u,
               5u,    7u,    9u,    4u,    4u,    5u,    6u,    4u,    6u,
               1u,    1u,    3u,    1u,    4u,    2u,    2u,    3u,    0u,
               0u,    1u,    1u,    0u,    1u,    0u,    0u,    0u,    0u,
               0u,    0u,    1u,    0u,    0u,    0u};
        EXPECT_VEC_EQ(expected_queued, queued);
    }
    else
    {
        cout << "No output saved for combination of RNG=\"" << celeritas_rng
             << "\", CLHEP=\"" << celeritas_clhep_version << "\", Geant4=\""
             << celeritas_geant4_version << "\"\n";
        PRINT_EXPECTED(active);
        PRINT_EXPECTED(queued);
    }

    // Check that callback was called
    EXPECT_EQ(active.size(), dummy_action_->num_execute_device());
    EXPECT_EQ(0, dummy_action_->num_execute_host());
}
