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
        EXPECT_EQ(381086,
                  result.cherenkov.total_num_photons
                      + result.scintillation.total_num_photons);

        EXPECT_EQ(38126, result.cherenkov.total_num_photons);
        EXPECT_EQ(78, result.cherenkov.num_photons.size());
        static unsigned int const expected_cherenkov_num_photons[] = {
            337u,  504u,  1609u, 1582u, 1314u, 1466u, 1164u, 880u, 777u,
            1477u, 1251u, 433u,  398u,  1273u, 271u,  1301u, 282u, 1132u,
            757u,  1132u, 869u,  246u,  604u,  135u,  515u,  45u,  452u,
            409u,  35u,   665u,  470u,  1040u, 339u,  523u,  526u, 219u,
            74u,   394u,  351u,  581u,  343u,  679u,  318u,  667u, 461u,
            608u,  213u,  550u,  228u,  528u,  160u,  485u,  334u, 480u,
            44u,   406u,  83u,   382u,  3u,    423u,  193u,  344u, 316u,
            248u,  265u,  41u,   232u,  210u,  37u,   124u,  154u, 53u,
            19u,   118u,  53u,   164u,  24u,   304u,
        };
        EXPECT_VEC_EQ(expected_cherenkov_num_photons,
                      result.cherenkov.num_photons);

        EXPECT_EQ(342960, result.scintillation.total_num_photons);
        EXPECT_EQ(188, result.scintillation.num_photons.size());
        static unsigned int const expected_scintillation_num_photons[] = {
            2826u,  3863u, 11346u, 11391u, 9567u, 10569u, 8310u, 6604u, 5701u,
            10383u, 8938u, 3409u,  2929u,  8819u, 2016u,  9815u, 2192u, 8369u,
            5504u,  8478u, 6466u,  1894u,  4402u, 823u,   3864u, 399u,  3423u,
            3072u,  202u,  4837u,  3463u,  7204u, 2365u,  4159u, 4377u, 1568u,
            537u,   2544u, 2682u,  4316u,  2846u, 4839u,  2672u, 4729u, 3384u,
            4449u,  2248u, 4125u,  2253u,  3731u, 2017u,  3818u, 2821u, 3469u,
            1861u,  3281u, 1983u,  3010u,  1670u, 3100u,  2266u, 2805u, 1177u,
            2665u,  1407u, 2303u,  389u,   2433u, 1875u,  2268u, 2241u, 1842u,
            1987u,  30u,   1982u,  1139u,  1641u, 1641u,  1090u, 1692u, 245u,
            1667u,  23u,   1185u,  285u,   33u,   233u,   17u,   320u,  20u,
            595u,   24u,   88u,    48u,    1137u, 53u,    604u,  28u,   243u,
            11u,    15u,   122u,   3u,     646u,  507u,   219u,  184u,  44u,
            17u,    27u,   1163u,  32u,    345u,  240u,   29u,   277u,  13u,
            51u,    68u,   62u,    455u,   62u,   54u,    19u,   2u,    56u,
            39u,    216u,  90u,    874u,   395u,  202u,   253u,  51u,   258u,
            15u,    218u,  1244u,  183u,   593u,  1443u,  6u,    1967u, 199u,
            3u,     1684u, 339u,   317u,   90u,   494u,   59u,   15u,   90u,
            494u,   611u,  693u,   286u,   56u,   17u,    150u,  664u,  836u,
            465u,   20u,   564u,   58u,    231u,  992u,   336u,  743u,  285u,
            58u,    112u,  1918u,  329u,   1060u, 1363u,  10u,   17u,   39u,
            722u,   12u,   1377u,  1808u,  2485u, 414u,   8u,    14u,
        };
        EXPECT_VEC_EQ(expected_scintillation_num_photons,
                      result.scintillation.num_photons);
    }
    else
    {
        EXPECT_EQ(41339, result.cherenkov.total_num_photons);
        EXPECT_EQ(88, result.cherenkov.num_photons.size());

        EXPECT_EQ(373102, result.scintillation.total_num_photons);
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
        EXPECT_EQ(94, result.accum.step_iters);
        EXPECT_EQ(1, result.accum.flushes);
        ASSERT_EQ(2, result.accum.generators.size());

        auto const& cherenkov = result.accum.generators[0];
        EXPECT_EQ(12, cherenkov.buffer_size);
        EXPECT_EQ(0, cherenkov.num_pending);
        EXPECT_EQ(5338, cherenkov.num_generated);

        auto const& scint = result.accum.generators[1];
        EXPECT_EQ(35, scint.buffer_size);
        EXPECT_EQ(0, scint.num_pending);
        EXPECT_EQ(50159, scint.num_generated);
    }
    EXPECT_EQ(0, result.scintillation.total_num_photons);
    EXPECT_EQ(0, result.cherenkov.total_num_photons);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
