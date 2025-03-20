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

#include "corecel/ScopedLogStorer.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/LogContextException.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/ModelImporter.hh"
#include "celeritas/optical/detail/OffloadParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"

#include "celeritas_test.hh"
#include "../LArSphereBase.hh"

using celeritas::detail::OpticalOffloadState;

namespace celeritas
{
namespace test
{
// TODO: replace this with explicit namespace importing
using namespace celeritas::optical;

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
        size_type num_photons{0};
        OffloadResult cherenkov;
        OffloadResult scintillation;

        // Step iteration at which the optical tracking loop launched
        size_type optical_launch_step{0};

        void print_expected() const;
    };

  public:
    void SetUp() override {}

    SPConstAction build_along_step() override;

    void build_optical_collector();

    VecPrimary make_primaries(size_type count);

    template<MemSpace M>
    RunResult run(size_type num_primaries,
                  size_type num_track_slots,
                  size_type num_steps);

  protected:
    using SizeId = ItemId<size_type>;
    using DistId = ItemId<GeneratorDistributionData>;
    using DistRange = ItemRange<GeneratorDistributionData>;

    // Optical collector options
    bool use_scintillation_{true};
    bool use_cherenkov_{true};
    size_type num_track_slots_{4096};
    size_type buffer_capacity_{256};
    size_type initializer_capacity_{8192};
    size_type auto_flush_{4096};
    units::MevEnergy primary_energy_{10.0};

    std::shared_ptr<OpticalCollector> collector_;
    StreamId stream_{0};
};

//---------------------------------------------------------------------------//
//! Print the expected result
void LArSphereOffloadTest::RunResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "EXPECT_EQ("
         << this->num_photons
         << ", result.num_photons);\n"
            "EXPECT_EQ("
         << this->cherenkov.total_num_photons
         << ", result.cherenkov.total_num_photons);\n"
            "EXPECT_EQ("
         << this->cherenkov.num_photons.size()
         << ", result.cherenkov.num_photons.size());\n"
            "static size_type const expected_cherenkov_num_photons[] = "
         << repr(this->cherenkov.num_photons)
         << ";\n"
            "EXPECT_VEC_EQ(expected_cherenkov_num_photons, "
            "result.cherenkov.num_photons);\n"
            "static real_type const expected_cherenkov_charge[] = "
         << repr(this->cherenkov.charge)
         << ";\n"
            "EXPECT_VEC_EQ(expected_cherenkov_charge, "
            "result.cherenkov.charge);\n"
            "EXPECT_EQ("
         << this->scintillation.total_num_photons
         << ", result.scintillation.total_num_photons);\n"
            "EXPECT_EQ("
         << this->scintillation.num_photons.size()
         << ", result.scintillation.num_photons.size());\n"
            "static size_type const expected_scintillation_num_photons[] = "
         << repr(this->scintillation.num_photons)
         << ";\n"
            "EXPECT_VEC_EQ(expected_scintillation_num_photons, "
            "result.scintillation.num_photons);\n"
            "static real_type const expected_scintillation_charge[] = "
         << repr(this->scintillation.charge)
         << ";\n"
            "EXPECT_VEC_EQ(expected_scintillation_charge, "
            "result.scintillation.charge);\n"
            "/*** END CODE ***/\n";
}

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
    OpticalCollector::Input inp;
    inp.material = this->optical_material();
    if (use_cherenkov_)
    {
        inp.cherenkov = this->cherenkov();
    }
    if (use_scintillation_)
    {
        inp.scintillation = this->scintillation();
    }
    inp.num_track_slots = num_track_slots_;
    inp.buffer_capacity = buffer_capacity_;
    inp.initializer_capacity = initializer_capacity_;
    inp.auto_flush = auto_flush_;

    using IMC = celeritas::optical::ImportModelClass;

    ModelImporter importer{
        this->imported_data(), this->optical_material(), this->material()};
    std::vector<IMC> imcs{IMC::absorption, IMC::rayleigh};
    for (auto imc : imcs)
    {
        if (auto builder = importer(imc))
        {
            inp.model_builders.push_back(*builder);
        }
    }

    collector_
        = std::make_shared<OpticalCollector>(*this->core(), std::move(inp));
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
    using DistRef
        = Collection<GeneratorDistributionData, Ownership::reference, M>;

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

    // Access the optical offload data
    auto const& offload_state = get<OpticalOffloadState<M>>(
        step.state().aux(), collector_->offload_aux_id());

    RunResult result;

    // Initial step
    auto primaries = this->make_primaries(num_primaries);
    StepperResult count;
    CELER_TRY_HANDLE(count = step(make_span(primaries)), log_context);

    size_type step_iter = 1;
    while (count && step_iter++ < num_steps)
    {
        if (!offload_state.buffer_size.num_photons)
        {
            result.optical_launch_step = step_iter;

            // TODO: For now abort immediately after primaries are generated
            // since the tracking loop hasn't been implemented
            break;
        }
        CELER_TRY_HANDLE(count = step(), log_context);
    }

    auto get_result = [&](OffloadResult& result,
                          DistRef const& buffer,
                          size_type size) {
        auto host_buffer = copy_to_host(buffer);
        std::set<real_type> charge;
        for (auto const& dist : host_buffer[DistRange(DistId(0), DistId(size))])
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

    auto const& state = offload_state.store.ref();
    auto const& sizes = offload_state.buffer_size;
    get_result(result.cherenkov, state.cherenkov, sizes.cherenkov);
    get_result(result.scintillation, state.scintillation, sizes.scintillation);
    result.num_photons = sizes.num_photons;

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
    auto_flush_ = size_type(-1);
    num_track_slots_ = 4;
    this->build_optical_collector();

    auto result = this->run<MemSpace::host>(4, num_track_slots_, 64);

    EXPECT_EQ(result.cherenkov.total_num_photons
                  + result.scintillation.total_num_photons,
              result.num_photons);

    static real_type const expected_cherenkov_charge[] = {-1, 1};
    EXPECT_VEC_EQ(expected_cherenkov_charge, result.cherenkov.charge);

    static real_type const expected_scintillation_charge[] = {-1, 0, 1};
    EXPECT_VEC_EQ(expected_scintillation_charge, result.scintillation.charge);

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(23471, result.cherenkov.total_num_photons);
        EXPECT_EQ(48, result.cherenkov.num_photons.size());
        static unsigned int const expected_cherenkov_num_photons[] = {
            337u, 503u,  1532u, 1485u, 788u, 610u, 1271u, 433u, 912u, 1051u,
            756u, 1124u, 796u,  854u,  446u, 420u, 582u,  648u, 704u, 825u,
            419u, 496u,  520u,  213u,  338u, 376u, 391u,  517u, 238u, 270u,
            254u, 370u,  23u,   115u,  129u, 317u, 183u,  10u,  3u,   416u,
            298u, 541u,  200u,  215u,  16u,  292u, 106u,  128u,
        };
        EXPECT_VEC_EQ(expected_cherenkov_num_photons,
                      result.cherenkov.num_photons);

        EXPECT_EQ(2101939, result.scintillation.total_num_photons);
        EXPECT_EQ(110, result.scintillation.num_photons.size());
        static unsigned int const expected_scintillation_num_photons[] = {
            27991u, 37559u, 114494u, 114637u, 58336u, 45280u, 90827u, 33901u,
            68555u, 74187u, 55095u,  83307u,  53825u, 59271u, 33295u, 30706u,
            42674u, 46522u, 48829u,  59030u,  33903u, 36690u, 38673u, 14395u,
            27169u, 29601u, 30544u,  39639u,  22299u, 23608u, 24698u, 29535u,
            18202u, 19521u, 20364u,  25475u,  10570u, 17164u, 17451u, 21131u,
            187u,   705u,   3100u,   17945u,  720u,   7048u,  11u,    6518u,
            1502u,  334u,   138u,    167u,    609u,   167u,   5209u,  4883u,
            1209u,  3268u,  1445u,   622u,    848u,   2250u,  778u,   3000u,
            11290u, 602u,   972u,    804u,    6603u,  8629u,  1217u,  9059u,
            13145u, 7969u,  8640u,   17523u,  146u,   4284u,  737u,   20u,
            8835u,  256u,   4210u,   152u,    1065u,  959u,   564u,   1485u,
            158u,   144u,   285u,    4449u,   173u,   155u,   33080u, 273u,
            1965u,  26445u, 38988u,  21405u,  20128u, 18024u, 27077u, 7972u,
            10375u, 144u,   20416u,  517u,    17255u, 1729u,
        };
        EXPECT_VEC_EQ(expected_scintillation_num_photons,
                      result.scintillation.num_photons);
    }
    else
    {
        EXPECT_EQ(20508, result.cherenkov.total_num_photons);
        EXPECT_EQ(53, result.cherenkov.num_photons.size());

        EXPECT_SOFT_EQ(
            2103651.0f,
            static_cast<float>(result.scintillation.total_num_photons));

        EXPECT_EQ(136, result.scintillation.num_photons.size());
    }
}

TEST_F(LArSphereOffloadTest, TEST_IF_CELER_DEVICE(device_distributions))
{
    auto_flush_ = size_type(-1);
    num_track_slots_ = 8;
    this->build_optical_collector();

    auto result = this->run<MemSpace::device>(8, num_track_slots_, 32);

    EXPECT_EQ(result.cherenkov.total_num_photons
                  + result.scintillation.total_num_photons,
              result.num_photons);

    static real_type const expected_cherenkov_charge[] = {-1, 1};
    EXPECT_VEC_EQ(expected_cherenkov_charge, result.cherenkov.charge);

    static real_type const expected_scintillation_charge[] = {-1, 0, 1};
    EXPECT_VEC_EQ(expected_scintillation_charge, result.scintillation.charge);

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(41811, result.cherenkov.total_num_photons);
        EXPECT_EQ(77, result.cherenkov.num_photons.size());
        static unsigned int const expected_cherenkov_num_photons[] = {
            337u, 503u,  1532u, 1485u, 1376u, 1471u, 1153u, 877u, 788u,
            610u, 1271u, 433u,  1068u, 1238u, 110u,  705u,  912u, 1051u,
            756u, 1124u, 779u,  1014u, 594u,  532u,  796u,  854u, 446u,
            420u, 639u,  747u,  354u,  610u,  582u,  648u,  704u, 825u,
            475u, 579u,  827u,  478u,  419u,  496u,  520u,  213u, 107u,
            472u, 712u,  324u,  338u,  376u,  391u,  517u,  6u,   372u,
            675u, 68u,   238u,  270u,  254u,  370u,  315u,  231u, 461u,
            61u,  23u,   115u,  129u,  317u,  188u,  97u,   406u, 183u,
            22u,  268u,  10u,   128u,  16u,
        };
        EXPECT_VEC_EQ(expected_cherenkov_num_photons,
                      result.cherenkov.num_photons);

        EXPECT_EQ(3641180, result.scintillation.total_num_photons);
        EXPECT_EQ(189, result.scintillation.num_photons.size());
        static unsigned int const expected_scintillation_num_photons[] = {
            27991u, 37559u, 114494u, 114637u, 95965u, 107707u, 83200u, 66412u,
            58336u, 45280u, 90827u,  33901u,  82348u, 92196u,  8026u,  50246u,
            68555u, 74187u, 55095u,  83307u,  54711u, 71651u,  42417u, 36917u,
            53825u, 59271u, 33295u,  30706u,  45810u, 56835u,  23107u, 43619u,
            42674u, 46522u, 48829u,  59030u,  36515u, 44766u,  58969u, 34673u,
            33903u, 36690u, 38673u,  14395u,  11257u, 35520u,  54220u, 27632u,
            27169u, 29601u, 30544u,  39639u,  273u,   28493u,  47830u, 5808u,
            22299u, 23608u, 24698u,  29535u,  26582u, 22501u,  37650u, 18666u,
            18202u, 19521u, 20364u,  25475u,  21847u, 19102u,  30255u, 13766u,
            10570u, 17164u, 17451u,  21131u,  18346u, 15207u,  24321u, 15580u,
            705u,   3100u,  17945u,  9017u,   19929u, 10489u,  7048u,  3967u,
            17347u, 5714u,  133u,    1440u,   6506u,  3452u,   1102u,  14u,
            11u,    1285u,  11258u,  1578u,   4608u,  6518u,   375u,   10850u,
            512u,   145u,   1463u,   8507u,   215u,   166u,    1478u,  100u,
            3052u,  9070u,  138u,    4129u,   767u,   338u,    6u,     5209u,
            1480u,  1211u,  868u,    3986u,   715u,   16323u,  151u,   124u,
            11609u, 12504u, 451u,    4627u,   2272u,  3743u,   2102u,  3072u,
            159u,   1311u,  3491u,   780u,    778u,   158u,    760u,   538u,
            1925u,  165u,   72u,     170u,    767u,   1157u,   174u,   7785u,
            140u,   6603u,  3835u,   16u,     1061u,  599u,    519u,   15u,
            3621u,  157u,   2766u,   152u,    825u,   139u,    1295u,  7753u,
            1170u,  11176u, 157u,    7690u,   576u,   527u,    8201u,  4391u,
            297u,   484u,   144u,    3106u,   351u,   2989u,   1664u,  6415u,
            4442u,  695u,   360u,    153u,    1683u,
        };
        EXPECT_VEC_EQ(expected_scintillation_num_photons,
                      result.scintillation.num_photons);
    }
    else
    {
        EXPECT_EQ(39110, result.cherenkov.total_num_photons);
        EXPECT_EQ(81, result.cherenkov.num_photons.size());

        EXPECT_EQ(3619371, result.scintillation.total_num_photons);
        EXPECT_EQ(200, result.scintillation.num_photons.size());
    }
}

TEST_F(LArSphereOffloadTest, cherenkov_distributiona)
{
    use_scintillation_ = false;
    auto_flush_ = size_type(-1);
    num_track_slots_ = 4;
    this->build_optical_collector();

    auto result = this->run<MemSpace::host>(4, num_track_slots_, 16);

    EXPECT_EQ(0, result.scintillation.total_num_photons);
    EXPECT_EQ(0, result.scintillation.num_photons.size());

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(19601, result.cherenkov.total_num_photons);
        EXPECT_EQ(37, result.cherenkov.num_photons.size());
    }
    else
    {
        EXPECT_EQ(20790, result.cherenkov.total_num_photons);
        EXPECT_EQ(43, result.cherenkov.num_photons.size());
    }
}

TEST_F(LArSphereOffloadTest, scintillation_distributions)
{
    use_cherenkov_ = false;
    auto_flush_ = size_type(-1);
    num_track_slots_ = 4;
    this->build_optical_collector();

    auto result = this->run<MemSpace::host>(4, num_track_slots_, 16);

    EXPECT_EQ(0, result.cherenkov.total_num_photons);
    EXPECT_EQ(0, result.cherenkov.num_photons.size());

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(1639326, result.scintillation.total_num_photons);
        EXPECT_EQ(53, result.scintillation.num_photons.size());
    }
    else
    {
        EXPECT_SOFT_EQ(
            1666806,
            static_cast<float>(result.scintillation.total_num_photons));
        EXPECT_EQ(52, result.scintillation.num_photons.size());
    }
}

TEST_F(LArSphereOffloadTest, host_generate_small)
{
    primary_energy_ = units::MevEnergy{0.01};
    num_track_slots_ = 32;
    buffer_capacity_ = 4096;
    initializer_capacity_ = 4096;
    auto_flush_ = 1;
    this->build_optical_collector();

    // Run with 2 core track slots and 32 optical track slots
    ScopedLogStorer scoped_log_{&celeritas::self_logger(), LogLevel::debug};
    auto result = this->run<MemSpace::host>(4, 2, 2);

    static char const* const expected_log_messages[] = {
        "Celeritas optical state initialization complete",
        "Celeritas core state initialization complete",
        "No Cherenkov photons to generate",
        "Generated 1028 Scintillation photons from 2 distributions",
        R"(Generated 1028 optical photons which completed 1028 total steps over 33 iterations)",
        "Deallocating host core state (stream 0)"};
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    }
    static char const* const expected_log_levels[]
        = {"status", "status", "debug", "debug", "debug", "debug"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

TEST_F(LArSphereOffloadTest, host_generate)
{
    num_track_slots_ = 262144;
    buffer_capacity_ = 1024;
    initializer_capacity_ = 524288;
    auto_flush_ = 16384;
    this->build_optical_collector();

    // Run with 512 core track slots and 2^18 optical track slots
    ScopedLogStorer scoped_log_{&celeritas::self_logger(), LogLevel::debug};
    auto result = this->run<MemSpace::host>(4, 512, 16);

    static char const* const expected_log_messages[] = {
        "Celeritas optical state initialization complete",
        "Celeritas core state initialization complete",
        "Generated 4258 Cherenkov photons from 4 distributions",
        "Generated 319935 Scintillation photons from 4 distributions",
        R"(Generated 324193 optical photons which completed 324193 total steps over 2 iterations)",
        "Deallocating host core state (stream 0)",
    };
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    }
    static char const* const expected_log_levels[]
        = {"status", "status", "debug", "debug", "debug", "debug"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());

    EXPECT_EQ(2, result.optical_launch_step);
    EXPECT_EQ(0, result.scintillation.total_num_photons);
    EXPECT_EQ(0, result.cherenkov.total_num_photons);
}

TEST_F(LArSphereOffloadTest, TEST_IF_CELER_DEVICE(device_generate))
{
    num_track_slots_ = 1024;
    buffer_capacity_ = 2048;
    initializer_capacity_ = 524288;
    auto_flush_ = 262144;
    this->build_optical_collector();

    ScopedLogStorer scoped_log_{&celeritas::self_logger()};
    auto result = this->run<MemSpace::device>(1, num_track_slots_, 16);
    static char const* const expected_log_levels[] = {"status", "status"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());

    EXPECT_EQ(7, result.optical_launch_step);
    EXPECT_EQ(0, result.scintillation.total_num_photons);
    EXPECT_EQ(0, result.cherenkov.total_num_photons);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
