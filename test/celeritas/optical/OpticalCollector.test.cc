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
#include "corecel/random/distribution/IsotropicDistribution.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "geocel/UnitUtils.hh"
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
constexpr bool reference_configuration
    = ((CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
       && (CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_VECGEOM
           || !CELERITAS_VECGEOM_SURFACE)
       && CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW);

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
        std::set<real_type> charge;
    };

    struct RunResult
    {
        // Optical distribution data
        OffloadResult cherenkov;
        OffloadResult scintillation;
        CounterAccumStats accum;
    };

  public:
    void SetUp() override
    {
        // Set default values
        input_.num_track_slots = 4096;
        input_.buffer_capacity = 512;
        input_.auto_flush = input_.num_track_slots;
        params_ = this->optical_params_input();
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
    optical::CoreParams::Input params_;
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
    inp.optical_params
        = std::make_shared<optical::CoreParams>(std::move(params_));
    collector_
        = std::make_shared<OpticalCollector>(*this->core(), std::move(inp));

    // Check accessors
    EXPECT_TRUE(collector_->optical_params());
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
    step_inp.actions = std::make_shared<ActionSequence>(
        *this->action_reg(), ActionSequence::Options{});
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
    auto gen_id = gen_reg.find("optical-generate");
    CELER_ASSERT(gen_id);

    // Access the auxiliary data for the generator
    auto const* aux
        = dynamic_cast<AuxParamsInterface const*>(gen_reg.at(gen_id).get());
    CELER_ASSERT(aux);
    auto const& state
        = get<optical::GeneratorState<M>>(step.state().aux(), aux->aux_id());
    auto buffer = copy_to_host(state.store.ref().distributions);

    for (auto const& dist :
         buffer[DistRange(DistId(state.counters.buffer_size))])
    {
        auto& offload_result = dist.type == GeneratorType::cherenkov
                                   ? result.cherenkov
                                   : result.scintillation;

        offload_result.total_num_photons += dist.num_photons;
        offload_result.num_photons.push_back(dist.num_photons);
        if (!dist)
        {
            continue;
        }
        offload_result.charge.insert(dist.charge.value());

        auto const& pre = dist.points[StepPoint::pre];
        auto const& post = dist.points[StepPoint::post];
        EXPECT_GT(pre.speed, zero_quantity());
        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            EXPECT_NE(post.pos, pre.pos);
        }
        EXPECT_GT(dist.step_length, 0);
        EXPECT_EQ(0, dist.material.get());
    }
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
    input_.auto_flush = std::numeric_limits<size_type>::max();
    input_.num_track_slots = 4;
    this->build_optical_collector();

    size_type primaries = 4;
    size_type core_track_slots = 4;
    // Even with an infinite auto flush, the optical launcher will flush when
    // there are no more active core tracks. Restrict the number of core step
    // iterations such that we can retrieve the distributions before flushing.
    size_type steps = 45;
    auto result = this->run<MemSpace::host>(primaries, core_track_slots, steps);

    // No steps ran
    EXPECT_EQ(0, result.accum.steps);
    EXPECT_EQ(0, result.accum.step_iters);
    EXPECT_EQ(0, result.accum.flushes);
    EXPECT_EQ(1, result.accum.generators.size());

    static real_type const expected_cherenkov_charge[] = {-1, 1};
    EXPECT_VEC_EQ(expected_cherenkov_charge, result.cherenkov.charge);

    static real_type const expected_scintillation_charge[] = {-1, 0, 1};
    EXPECT_VEC_EQ(expected_scintillation_charge, result.scintillation.charge);

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
        && (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW))
    {
        EXPECT_EQ(228154,
                  result.cherenkov.total_num_photons
                      + result.scintillation.total_num_photons);

        EXPECT_EQ(21304, result.cherenkov.total_num_photons);
        static unsigned int const expected_cherenkov_num_photons[] = {
            371u, 539u,  1609u, 1582u, 1095u, 530u, 1251u, 481u, 892u, 355u,
            757u, 1129u, 239u,  376u,  428u,  441u, 610u,  698u, 589u, 222u,
            482u, 159u,  38u,   236u,  345u,  822u, 303u,  108u, 17u,  631u,
            211u, 62u,   502u,  28u,   392u,  281u, 119u,  7u,   11u,  194u,
            19u,  372u,  270u,  118u,  32u,   4u,   533u,  395u, 268u, 151u,
        };
        EXPECT_VEC_EQ(expected_cherenkov_num_photons,
                      result.cherenkov.num_photons);

        EXPECT_EQ(206850, result.scintillation.total_num_photons);
        static unsigned int const expected_scintillation_num_photons[] = {
            2678u, 3867u, 11346u, 11391u, 7849u, 3835u, 8938u, 3409u, 6309u,
            2820u, 5504u, 8457u,  1923u,  2355u, 3423u, 3099u, 4546u, 5263u,
            4175u, 1929u, 3525u,  1106u,  341u,  2307u, 2879u, 5919u, 2679u,
            1900u, 219u,  4698u,  2228u,  1695u, 1796u, 3696u, 1798u, 1303u,
            3008u, 960u,  974u,   2403u,  644u,  1932u, 109u,  1765u, 243u,
            11u,   200u,  1251u,  519u,   384u,  15u,   654u,  560u,  1060u,
            180u,  64u,   167u,   117u,   19u,   116u,  149u,  599u,  29u,
            15u,   1626u, 16u,    310u,   1785u, 316u,  15u,   618u,  215u,
            348u,  1782u, 610u,   2210u,  728u,  368u,  1797u, 18u,   68u,
            888u,  38u,   109u,   1709u,  77u,   87u,   2958u, 20u,   19u,
            2376u, 255u,  154u,   1997u,  369u,  54u,   1775u, 1812u, 13u,
            120u,  1027u, 59u,    18u,    43u,   487u,  9u,    21u,   12u,
            273u,  643u,  137u,   1795u,  66u,   116u,  607u,  11u,   199u,
            20u,   147u,  106u,   11u,    45u,   248u,  492u,  3951u, 1176u,
            3068u, 4u,    2446u,  212u,   2039u, 350u,
        };
        EXPECT_VEC_EQ(expected_scintillation_num_photons,
                      result.scintillation.num_photons);
    }
}

TEST_F(LArSphereOffloadTest, TEST_IF_CELER_DEVICE(device_distributions))
{
    input_.auto_flush = std::numeric_limits<size_type>::max();
    input_.num_track_slots = 8;
    this->build_optical_collector();

    size_type primaries = 8;
    size_type core_track_slots = 8;
    size_type steps = 32;
    auto result
        = this->run<MemSpace::device>(primaries, core_track_slots, steps);

    // No steps ran
    EXPECT_EQ(0, result.accum.steps);
    EXPECT_EQ(0, result.accum.step_iters);
    EXPECT_EQ(0, result.accum.flushes);
    EXPECT_EQ(1, result.accum.generators.size());

    static real_type const expected_cherenkov_charge[] = {-1, 1};
    EXPECT_VEC_EQ(expected_cherenkov_charge, result.cherenkov.charge);

    static real_type const expected_scintillation_charge[] = {-1, 0, 1};
    EXPECT_VEC_EQ(expected_scintillation_charge, result.scintillation.charge);

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
        && CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW)
    {
        EXPECT_EQ(416523,
                  result.cherenkov.total_num_photons
                      + result.scintillation.total_num_photons);

        EXPECT_EQ(41328, result.cherenkov.total_num_photons);
        EXPECT_EQ(88, result.cherenkov.num_photons.size());
        static unsigned int const expected_cherenkov_num_photons[] = {
            371u, 539u,  1609u, 1582u, 1328u, 1451u, 1084u, 894u, 1095u,
            530u, 1251u, 481u,  399u,  1167u, 265u,  1328u, 892u, 355u,
            757u, 1129u, 914u,  941u,  640u,  74u,   239u,  376u, 428u,
            441u, 929u,  831u,  477u,  1117u, 610u,  698u,  589u, 222u,
            756u, 504u,  670u,  498u,  482u,  159u,  38u,   236u, 651u,
            49u,  502u,  428u,  345u,  822u,  303u,  108u,  323u, 473u,
            139u, 336u,  17u,   631u,  211u,  301u,  331u,  128u, 453u,
            62u,  502u,  28u,   190u,  211u,  5u,    326u,  392u, 9u,
            33u,  90u,   281u,  175u,  119u,  14u,   15u,   36u,  219u,
            29u,  22u,   2u,    91u,   332u,  179u,  39u,
        };
        EXPECT_VEC_EQ(expected_cherenkov_num_photons,
                      result.cherenkov.num_photons);

        EXPECT_EQ(375195, result.scintillation.total_num_photons);
        EXPECT_EQ(196, result.scintillation.num_photons.size());
        static unsigned int const expected_scintillation_num_photons[] = {
            2678u, 3867u, 11346u, 11391u, 9567u, 10882u, 8310u,  6604u, 7849u,
            3835u, 8938u, 3409u,  2923u,  8957u, 1886u,  10021u, 6309u, 2820u,
            5504u, 8457u, 6232u,  6883u,  4777u, 569u,   1923u,  2355u, 3423u,
            3099u, 6973u, 5490u,  3387u,  7760u, 4546u,  5263u,  4175u, 1929u,
            5568u, 3973u, 4469u,  3610u,  3525u, 1106u,  341u,   2307u, 4386u,
            306u,  3658u, 2944u,  2879u,  5919u, 2679u,  1900u,  2343u, 3457u,
            1064u, 2522u, 219u,   4698u,  2228u, 1695u,  2562u,  2685u, 1077u,
            3493u, 1796u, 3696u,  1798u,  2117u, 2124u,  1780u,  2758u, 1303u,
            3008u, 960u,  1759u,  1840u,  549u,  918u,   1769u,  2403u, 1084u,
            673u,  1151u, 640u,   2034u,  392u,  1932u,  115u,   1778u, 1552u,
            1765u, 11u,   1501u,  602u,   261u,  200u,   519u,   223u,  1637u,
            14u,   196u,  15u,    527u,   25u,   51u,    217u,   200u,  170u,
            13u,   21u,   16u,    335u,   12u,   7u,     409u,   118u,  389u,
            16u,   422u,  17u,    11u,    193u,  132u,   206u,   18u,   15u,
            120u,  32u,   415u,   103u,   185u,  12u,    310u,   220u,  1139u,
            36u,   1548u, 240u,   39u,    15u,   1745u,  1013u,  282u,  513u,
            382u,  1569u, 13u,    72u,    646u,  12u,    17u,    50u,   954u,
            918u,  147u,  492u,   790u,   1977u, 13u,    171u,   534u,  21u,
            367u,  1102u, 368u,   16u,    492u,  622u,   187u,   16u,   2154u,
            344u,  1797u, 151u,   44u,    44u,   1889u,  24u,    746u,  73u,
            63u,   883u,  1946u,  171u,   23u,   329u,   2634u,  1587u, 16u,
            49u,   192u,  2151u,  1700u,  1817u, 39u,    51u,
        };
        EXPECT_VEC_EQ(expected_scintillation_num_photons,
                      result.scintillation.num_photons);
    }
}

TEST_F(LArSphereOffloadTest, cherenkov_distributiona)
{
    params_.scintillation = nullptr;
    input_.auto_flush = std::numeric_limits<size_type>::max();
    input_.num_track_slots = 4;
    this->build_optical_collector();

    size_type primaries = 4;
    size_type core_track_slots = 4;
    size_type steps = 16;
    auto result = this->run<MemSpace::host>(primaries, core_track_slots, steps);

    EXPECT_EQ(0, result.scintillation.total_num_photons);
    EXPECT_EQ(0, result.scintillation.num_photons.size());

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
        && CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW)
    {
        EXPECT_EQ(19970, result.cherenkov.total_num_photons);
        EXPECT_EQ(38, result.cherenkov.num_photons.size());
    }
}

TEST_F(LArSphereOffloadTest, scintillation_distributions)
{
    params_.cherenkov = nullptr;
    input_.auto_flush = std::numeric_limits<size_type>::max();
    input_.num_track_slots = 4;
    this->build_optical_collector();

    size_type primaries = 4;
    size_type core_track_slots = 4;
    size_type steps = 16;
    auto result = this->run<MemSpace::host>(primaries, core_track_slots, steps);

    EXPECT_EQ(0, result.cherenkov.total_num_photons);
    EXPECT_EQ(0, result.cherenkov.num_photons.size());

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
        && CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW)
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

    if (reference_configuration)
    {
        constexpr unsigned int expected_steps = 117;
        constexpr unsigned int expected_step_iters = 5;
        EXPECT_EQ(expected_steps, result.accum.steps);
        EXPECT_EQ(expected_step_iters, result.accum.step_iters);
        EXPECT_EQ(1, result.accum.flushes);
        ASSERT_EQ(1, result.accum.generators.size());

        auto const& generator = result.accum.generators[0];
        EXPECT_EQ(2, generator.buffer_size);
        EXPECT_EQ(0, generator.num_pending);
        EXPECT_EQ(109, generator.num_generated);
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

    if (reference_configuration)
    {
        unsigned int expected_steps = 19283;
        unsigned int expected_step_iters = 3;
        EXPECT_EQ(expected_steps, static_cast<double>(result.accum.steps));
        EXPECT_EQ(expected_step_iters, result.accum.step_iters);
        EXPECT_EQ(1, result.accum.flushes);
        ASSERT_EQ(1, result.accum.generators.size());

        auto const& generator = result.accum.generators[0];
        EXPECT_EQ(7, generator.buffer_size);
        EXPECT_EQ(0, generator.num_pending);
        EXPECT_EQ(18214, generator.num_generated);

        EXPECT_EQ(2033, result.scintillation.total_num_photons);
        EXPECT_EQ(273, result.cherenkov.total_num_photons);
    }
    else if (CELERITAS_REAL_TYPE != CELERITAS_REAL_TYPE_DOUBLE)
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

    if (reference_configuration)
    {
        constexpr int ref_steps = 59;
        EXPECT_EQ(ref_steps, result.accum.step_iters);
        EXPECT_EQ(1, result.accum.flushes);
        ASSERT_EQ(1, result.accum.generators.size());

        auto const& generator = result.accum.generators[0];
        EXPECT_EQ(50, generator.buffer_size);
        EXPECT_EQ(0, generator.num_pending);
        EXPECT_EQ(55684, generator.num_generated);
    }
    EXPECT_EQ(0, result.scintillation.total_num_photons);
    EXPECT_EQ(0, result.cherenkov.total_num_photons);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
