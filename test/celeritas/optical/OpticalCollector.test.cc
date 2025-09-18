//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/OpticalCollector.hh"

#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/LogContextException.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/random/IsotropicDistribution.hh"
#include "celeritas/LArSphereBase.hh"
#include "celeritas/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/inp/Field.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/gen/GeneratorData.hh"
#include "celeritas/phys/GeneratorRegistry.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class LArSphereOffloadTest : public LArSphereBase
{
  public:
    using VecPrimary = std::vector<celeritas::Primary>;

    struct OffloadResult
    {
        size_type total_num_photons{0};
        std::vector<size_type> num_photons;
        std::vector<real_type> charge;
    };

    struct RunResult
    {
        // Optical distribution data
        OffloadResult cherenkov;
        OffloadResult scintillation;
        OpticalAccumStats accum;
    };

  public:
    void SetUp() override
    {
        // Set default values
        input_.optical_params = this->optical_params();
        input_.cherenkov = this->cherenkov();
        input_.scintillation = this->scintillation();
        input_.num_track_slots = 4096;
        input_.buffer_capacity = 256;
        input_.auto_flush = 4096;
    }

    SPConstAction build_along_step() override;

    void build_optical_collector();

    VecPrimary make_primaries(size_type count);

    template<MemSpace M>
    RunResult run(size_type num_primaries,
                  size_type num_track_slots,
                  size_type num_steps);

  protected:
    using SizeId = ItemId<size_type>;
    using DistId = ItemId<optical::GeneratorDistributionData>;
    using DistRange = ItemRange<optical::GeneratorDistributionData>;

    units::MevEnergy primary_energy_{10.0};

    OpticalCollector::Input input_;
    std::shared_ptr<OpticalCollector> collector_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct along-step action.
 */
auto LArSphereOffloadTest::build_along_step() -> SPConstAction
{
    auto& action_reg = *this->action_reg();
    UniformFieldParams::Input field_inp;
    field_inp.strength = {0, 0, 1};
    auto msc = UrbanMscParams::from_import(
        *this->particle(), *this->material(), this->imported_data());

    auto result = std::make_shared<AlongStepUniformMscAction>(
        action_reg.next_id(), *this->geometry(), field_inp, nullptr, msc);
    CELER_ASSERT(result);
    CELER_ASSERT(result->has_msc());
    action_reg.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct optical collector.
 */
void LArSphereOffloadTest::build_optical_collector()
{
    OpticalCollector::Input inp = input_;
    collector_
        = std::make_shared<OpticalCollector>(*this->core(), std::move(inp));

    // Check accessors
    EXPECT_TRUE(collector_->optical_params());
    EXPECT_EQ(static_cast<bool>(input_.cherenkov),
              static_cast<bool>(collector_->cherenkov()));
    EXPECT_EQ(static_cast<bool>(input_.scintillation),
              static_cast<bool>(collector_->scintillation()));
}

//---------------------------------------------------------------------------//
/*!
 * Generate a vector of primary particles.
 */
auto LArSphereOffloadTest::make_primaries(size_type count) -> VecPrimary
{
    celeritas::Primary p;
    p.event_id = EventId{0};
    p.energy = primary_energy_;
    p.position = from_cm(Real3{0, 0, 0});
    p.time = 0;

    Array<ParticleId, 2> const particles = {
        this->particle()->find(pdg::electron()),
        this->particle()->find(pdg::positron()),
    };
    CELER_ASSERT(particles[0] && particles[1]);

    std::vector<celeritas::Primary> result(count, p);
    IsotropicDistribution<> sample_dir;
    std::mt19937 rng;

    for (auto i : range(count))
    {
        result[i].direction = sample_dir(rng);
        result[i].particle_id = particles[i % particles.size()];
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Run a number of tracks.
 */
template<MemSpace M>
auto LArSphereOffloadTest::run(size_type num_primaries,
                               size_type num_track_slots,
                               size_type num_steps) -> RunResult
{
    if constexpr (M == MemSpace::device)
    {
        device().create_streams(1);
    }

    // Create the core stepper
    StepperInput step_inp;
    step_inp.params = this->core();
    step_inp.stream_id = StreamId{0};
    step_inp.num_track_slots = num_track_slots;
    Stepper<M> step(step_inp);
    LogContextException log_context{this->output_reg().get()};

    RunResult result;

    // Initial step
    auto primaries = this->make_primaries(num_primaries);
    StepperResult count;
    CELER_TRY_HANDLE(count = step(make_span(primaries)), log_context);

    size_type step_iter = 1;
    while (count && step_iter++ < num_steps)
    {
        CELER_TRY_HANDLE(count = step(), log_context);
    }

    auto const& gen_reg = collector_->gen_reg();

    auto get_result = [&](OffloadResult& result, GeneratorId gen_id) {
        if (!gen_id)
        {
            return;
        }
        // Access the auxiliary data for this generator
        auto const* aux = dynamic_cast<AuxParamsInterface const*>(
            gen_reg.at(gen_id).get());
        CELER_ASSERT(aux);
        auto const& state = get<optical::GeneratorState<M>>(step.state().aux(),
                                                            aux->aux_id());
        auto buffer = copy_to_host(state.store.ref().distributions);

        std::set<real_type> charge;
        for (auto const& dist :
             buffer[DistRange(DistId(state.counters.buffer_size))])
        {
            result.total_num_photons += dist.num_photons;
            result.num_photons.push_back(dist.num_photons);
            if (!dist)
            {
                continue;
            }
            charge.insert(dist.charge.value());

            auto const& pre = dist.points[StepPoint::pre];
            auto const& post = dist.points[StepPoint::post];
            EXPECT_GT(pre.speed, zero_quantity());
            EXPECT_NE(post.pos, pre.pos);
            EXPECT_GT(dist.step_length, 0);
            EXPECT_EQ(0, dist.material.get());
        }
        result.charge.insert(result.charge.end(), charge.begin(), charge.end());
    };

    // Access the optical offload data
    get_result(result.cherenkov, gen_reg.find("cherenkov-generate"));
    get_result(result.scintillation, gen_reg.find("scintillation-generate"));
    result.accum = collector_->exchange_counters(step.sp_state()->aux());

    return result;
}

//---------------------------------------------------------------------------//
template LArSphereOffloadTest::RunResult
    LArSphereOffloadTest::run<MemSpace::host>(size_type, size_type, size_type);
template LArSphereOffloadTest::RunResult
    LArSphereOffloadTest::run<MemSpace::device>(size_type, size_type, size_type);

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(LArSphereOffloadTest, host_distributions)
{
    input_.max_step_iters = 0;
    input_.num_track_slots = 4;
    this->build_optical_collector();

    size_type primaries = 4;
    size_type core_track_slots = 4;
    size_type steps = 64;
    auto result = this->run<MemSpace::host>(primaries, core_track_slots, steps);

    EXPECT_EQ(2, result.accum.generators.size());

    static real_type const expected_cherenkov_charge[] = {-1, 1};
    EXPECT_VEC_EQ(expected_cherenkov_charge, result.cherenkov.charge);

    static real_type const expected_scintillation_charge[] = {-1, 0, 1};
    EXPECT_VEC_EQ(expected_scintillation_charge, result.scintillation.charge);

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(231683,
                  result.cherenkov.total_num_photons
                      + result.scintillation.total_num_photons);

        EXPECT_EQ(21963, result.cherenkov.total_num_photons);
        static unsigned int const expected_cherenkov_num_photons[] = {
            337ul,  504ul, 1609ul, 1582ul, 777ul, 1477ul, 1251ul, 433ul, 282ul,
            1132ul, 757ul, 1132ul, 515ul,  45ul,  452ul,  409ul,  339ul, 523ul,
            526ul,  219ul, 343ul,  679ul,  318ul, 667ul,  228ul,  528ul, 160ul,
            485ul,  83ul,  382ul,  3ul,    423ul, 248ul,  265ul,  124ul, 124ul,
            154ul,  288ul, 173ul,  14ul,   4ul,   5ul,    19ul,   303ul, 181ul,
            13ul,   102ul, 553ul,  106ul,  401ul, 224ul,  62ul,
        };
        EXPECT_VEC_EQ(expected_cherenkov_num_photons,
                      result.cherenkov.num_photons);

        EXPECT_EQ(209720, result.scintillation.total_num_photons);
        static unsigned int const expected_scintillation_num_photons[] = {
            2826ul, 3863ul, 11346ul, 11391ul, 5701ul, 10383ul, 8938ul, 3409ul,
            2192ul, 8369ul, 5504ul,  8478ul,  3864ul, 399ul,   3423ul, 3072ul,
            2365ul, 4159ul, 4377ul,  1568ul,  2846ul, 4839ul,  2672ul, 4729ul,
            2253ul, 3731ul, 2017ul,  3818ul,  1983ul, 3010ul,  1670ul, 3100ul,
            1407ul, 2303ul, 389ul,   2433ul,  1988ul, 1987ul,  13ul,   1982ul,
            1716ul, 1692ul, 726ul,   1667ul,  115ul,  233ul,   2455ul, 320ul,
            11ul,   2053ul, 1754ul,  1716ul,  755ul,  484ul,   240ul,  474ul,
            11ul,   206ul,  22ul,    646ul,   1249ul, 372ul,   199ul,  28ul,
            1047ul, 16ul,   226ul,   29ul,    65ul,   1836ul,  59ul,   976ul,
            1399ul, 7ul,    15ul,    33ul,    51ul,   199ul,   420ul,  11ul,
            2594ul, 159ul,  33ul,    2106ul,  481ul,  1694ul,  14ul,   1852ul,
            216ul,  68ul,   696ul,   867ul,   11ul,   1986ul,  247ul,  70ul,
            1618ul, 41ul,   18ul,    21ul,    5ul,    89ul,    7ul,    18ul,
            34ul,   64ul,   405ul,   12ul,    16ul,   3870ul,  32ul,   139ul,
            232ul,  861ul,  2799ul,  2274ul,  72ul,   14ul,    1959ul, 284ul,
            1412ul,
        };
        EXPECT_VEC_EQ(expected_scintillation_num_photons,
                      result.scintillation.num_photons);
    }
}

TEST_F(LArSphereOffloadTest, TEST_IF_CELER_DEVICE(device_distributions))
{
    input_.max_step_iters = 0;
    input_.num_track_slots = 8;
    this->build_optical_collector();

    size_type primaries = 8;
    size_type core_track_slots = 8;
    size_type steps = 32;
    auto result
        = this->run<MemSpace::device>(primaries, core_track_slots, steps);

    EXPECT_EQ(2, result.accum.generators.size());

    static real_type const expected_cherenkov_charge[] = {-1, 1};
    EXPECT_VEC_EQ(expected_cherenkov_charge, result.cherenkov.charge);

    static real_type const expected_scintillation_charge[] = {-1, 0, 1};
    EXPECT_VEC_EQ(expected_scintillation_charge, result.scintillation.charge);

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(3562272,
                  result.cherenkov.total_num_photons
                      + result.scintillation.total_num_photons);

        EXPECT_EQ(38699, result.cherenkov.total_num_photons);
        EXPECT_EQ(82, result.cherenkov.num_photons.size());
        static unsigned int const expected_cherenkov_num_photons[] = {
            337u,  504u,  1609u, 1582u, 1314u, 1466u, 1164u, 880u, 777u,
            1477u, 1251u, 433u,  398u,  1273u, 271u,  1301u, 282u, 1132u,
            757u,  1132u, 869u,  246u,  604u,  135u,  515u,  45u,  452u,
            409u,  35u,   665u,  470u,  1040u, 339u,  523u,  526u, 219u,
            74u,   394u,  351u,  581u,  343u,  679u,  318u,  667u, 461u,
            608u,  213u,  550u,  228u,  528u,  160u,  485u,  334u, 480u,
            44u,   406u,  83u,   382u,  3u,    423u,  193u,  344u, 316u,
            248u,  265u,  41u,   232u,  210u,  37u,   124u,  154u, 53u,
            19u,   118u,  302u,  162u,  162u,  1u,    301u,  27u,  3u,
            160u,
        };
        EXPECT_VEC_EQ(expected_cherenkov_num_photons,
                      result.cherenkov.num_photons);

        EXPECT_EQ(3523573, result.scintillation.total_num_photons);
        EXPECT_EQ(191, result.scintillation.num_photons.size());
        static unsigned int const expected_scintillation_num_photons[] = {
            27991u, 38157u,  114070u, 114477u, 95923u, 106832u, 82858u, 66126u,
            57893u, 103619u, 90287u,  33901u,  28955u, 88563u,  20427u, 98050u,
            21827u, 83989u,  55095u,  84026u,  63443u, 18590u,  43852u, 8258u,
            38355u, 3894u,   33219u,  30807u,  2363u,  48420u,  34733u, 71951u,
            24182u, 41506u,  43246u,  15732u,  5282u,  25256u,  27512u, 43955u,
            28341u, 47956u,  26749u,  47994u,  34426u, 44063u,  22581u, 41383u,
            22830u, 37627u,  20074u,  38203u,  27903u, 34719u,  18699u, 32830u,
            19233u, 30026u,  17229u,  30547u,  22604u, 27778u,  11994u, 26337u,
            13618u, 23721u,  4019u,   24306u,  18720u, 22417u,  21616u, 18459u,
            19787u, 317u,    19892u,  11747u,  16520u, 16590u,  10760u, 17217u,
            2425u,  17185u,  183u,    12042u,  2856u,  313u,    2398u,  162u,
            3256u,  181u,    6066u,   241u,    1003u,  579u,    11401u, 502u,
            6016u,  269u,    2524u,   143u,    157u,   1274u,   18u,    6273u,
            5298u,  1995u,   1958u,   408u,    161u,   327u,    11457u, 257u,
            3367u,  2349u,   294u,    2716u,   215u,   478u,    700u,   638u,
            4677u,  667u,    544u,    170u,    16u,    683u,    378u,   2103u,
            1048u,  8963u,   3773u,   2223u,   2446u,  510u,    2535u,  155u,
            2303u,  12407u,  1867u,   6069u,   14643u, 68u,     19850u, 2063u,
            13u,    17011u,  2461u,   3346u,   1034u,  4914u,   2982u,  155u,
            864u,   3400u,   7123u,   7028u,   3053u,  643u,    172u,   161u,
            1031u,  153u,    6319u,   3720u,   1003u,  5885u,   5529u,  176u,
            4786u,  34u,     25324u,  20567u,  3106u,  15352u,  13363u, 13176u,
            2050u,  20835u,  17342u,  158u,    157u,   26088u,  18297u, 38u,
            17647u, 3953u,   339u,    922u,    21243u, 9972u,   948u,
        };
        EXPECT_VEC_EQ(expected_scintillation_num_photons,
                      result.scintillation.num_photons);
    }
    else
    {
        EXPECT_EQ(41814, result.cherenkov.total_num_photons);
        EXPECT_EQ(89, result.cherenkov.num_photons.size());

        EXPECT_EQ(3773161, result.scintillation.total_num_photons);
        EXPECT_EQ(202, result.scintillation.num_photons.size());
    }
}

TEST_F(LArSphereOffloadTest, cherenkov_distributiona)
{
    input_.scintillation = nullptr;
    input_.max_step_iters = 0;
    input_.num_track_slots = 4;
    this->build_optical_collector();

    size_type primaries = 4;
    size_type core_track_slots = 4;
    size_type steps = 16;
    auto result = this->run<MemSpace::host>(primaries, core_track_slots, steps);

    EXPECT_EQ(0, result.scintillation.total_num_photons);
    EXPECT_EQ(0, result.scintillation.num_photons.size());

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(21060, result.cherenkov.total_num_photons);
        EXPECT_EQ(39, result.cherenkov.num_photons.size());
    }
    else
    {
        EXPECT_EQ(16454, result.cherenkov.total_num_photons);
        EXPECT_EQ(32, result.cherenkov.num_photons.size());
    }
}

TEST_F(LArSphereOffloadTest, scintillation_distributions)
{
    input_.cherenkov = nullptr;
    input_.max_step_iters = 0;
    input_.num_track_slots = 4;
    this->build_optical_collector();

    size_type primaries = 4;
    size_type core_track_slots = 4;
    size_type steps = 16;
    auto result = this->run<MemSpace::host>(primaries, core_track_slots, steps);

    EXPECT_EQ(0, result.cherenkov.total_num_photons);
    EXPECT_EQ(0, result.cherenkov.num_photons.size());

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(167469, result.scintillation.total_num_photons);
        EXPECT_EQ(48, result.scintillation.num_photons.size());

        // No steps ran
        EXPECT_EQ(0, result.accum.steps);
        EXPECT_EQ(0, result.accum.step_iters);
        EXPECT_EQ(0, result.accum.flushes);
        ASSERT_EQ(1, result.accum.generators.size());

        auto const& scint = result.accum.generators.front();
        EXPECT_EQ(0, scint.buffer_size);
        EXPECT_EQ(0, scint.num_pending);
        EXPECT_EQ(0, scint.num_generated);
    }
}

TEST_F(LArSphereOffloadTest, host_generate_small)
{
    input_.num_track_slots = 32;
    input_.buffer_capacity = 4096;
    input_.auto_flush = 1;
    this->build_optical_collector();

    primary_energy_ = units::MevEnergy{0.01};

    size_type primaries = 4;
    size_type core_track_slots = 2;
    size_type steps = 2;
    auto result = this->run<MemSpace::host>(primaries, core_track_slots, steps);

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(187, result.accum.steps);
        EXPECT_EQ(9, result.accum.step_iters);
        EXPECT_EQ(1, result.accum.flushes);
        ASSERT_EQ(2, result.accum.generators.size());

        auto const& cherenkov = result.accum.generators[0];
        EXPECT_EQ(0, cherenkov.buffer_size);
        EXPECT_EQ(0, cherenkov.num_pending);
        EXPECT_EQ(0, cherenkov.num_generated);

        auto const& scint = result.accum.generators[1];
        EXPECT_EQ(2, scint.buffer_size);
        EXPECT_EQ(0, scint.num_pending);
        EXPECT_EQ(109, scint.num_generated);
    }
}

TEST_F(LArSphereOffloadTest, host_generate)
{
    input_.num_track_slots = 262144;
    input_.buffer_capacity = 1024;
    input_.auto_flush = 16384;
    this->build_optical_collector();

    size_type primaries = 1;
    size_type core_track_slots = 512;
    size_type steps = 4;
    auto result = this->run<MemSpace::host>(primaries, core_track_slots, steps);

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_SOFT_NEAR(38115, static_cast<double>(result.accum.steps), 1e-4);
        EXPECT_EQ(11, result.accum.step_iters);
        EXPECT_EQ(1, result.accum.flushes);
        ASSERT_EQ(2, result.accum.generators.size());

        auto const& cherenkov = result.accum.generators[0];
        EXPECT_EQ(3, cherenkov.buffer_size);
        EXPECT_EQ(0, cherenkov.num_pending);
        EXPECT_EQ(2865, cherenkov.num_generated);

        auto const& scint = result.accum.generators[1];
        EXPECT_EQ(4, scint.buffer_size);
        EXPECT_EQ(0, scint.num_pending);
        EXPECT_EQ(20777, scint.num_generated);

        EXPECT_EQ(7227, result.scintillation.total_num_photons);
        EXPECT_EQ(970, result.cherenkov.total_num_photons);
    }
    else
    {
        EXPECT_GT(result.accum.step_iters, 0);
        EXPECT_GT(result.accum.flushes, 0);
        EXPECT_GT(result.scintillation.total_num_photons, 0);
        EXPECT_GT(result.cherenkov.total_num_photons, 0);
    }
}

TEST_F(LArSphereOffloadTest, TEST_IF_CELER_DEVICE(device_generate))
{
    input_.num_track_slots = 1024;
    input_.buffer_capacity = 2048;
    input_.auto_flush = 262144;
    this->build_optical_collector();

    size_type primaries = 1;
    size_type core_track_slots = 1024;
    size_type steps = 16;
    auto result
        = this->run<MemSpace::device>(primaries, core_track_slots, steps);

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(818, result.accum.step_iters);
        EXPECT_EQ(2, result.accum.flushes);
        ASSERT_EQ(2, result.accum.generators.size());

        auto const& cherenkov = result.accum.generators[0];
        EXPECT_EQ(12, cherenkov.buffer_size);
        EXPECT_EQ(0, cherenkov.num_pending);
        EXPECT_EQ(5338, cherenkov.num_generated);

        auto const& scint = result.accum.generators[1];
        EXPECT_EQ(35, scint.buffer_size);
        EXPECT_EQ(0, scint.num_pending);
        EXPECT_EQ(500472, scint.num_generated);
    }
    EXPECT_EQ(0, result.scintillation.total_num_photons);
    EXPECT_EQ(0, result.cherenkov.total_num_photons);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
