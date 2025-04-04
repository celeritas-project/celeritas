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
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/ModelImporter.hh"
#include "celeritas/optical/detail/OffloadParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "celeritas_test.hh"

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
        OpticalAccumStats accum;

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
            "EXPECT_EQ("
         << this->accum.steps
         << ", result.accum.steps);\n"
            "EXPECT_EQ("
         << this->accum.step_iters
         << ", result.accum.step_iters);\n"
            "EXPECT_EQ("
         << this->accum.flushes
         << ", result.accum.flushes);\n"
            "EXPECT_EQ("
         << this->accum.generators.cherenkov
         << ", result.accum.generators.cherenkov);\n"
            "EXPECT_EQ("
         << this->accum.generators.scintillation
         << ", result.accum.generators.scintillation);\n"
            "EXPECT_EQ("
         << this->accum.generators.photons
         << ", result.accum.generators.photons);\n"
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
        if (!offload_state.buffer_size.photons)
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
    result.num_photons = sizes.photons;

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
        EXPECT_EQ(21541, result.cherenkov.total_num_photons);
        EXPECT_EQ(53, result.cherenkov.num_photons.size());
        static unsigned int const expected_cherenkov_num_photons[] = {
            337u,  504u, 1609u, 1582u, 777u, 1477u, 1251u, 433u, 282u,
            1132u, 757u, 1132u, 515u,  45u,  452u,  409u,  339u, 523u,
            526u,  219u, 343u,  679u,  318u, 667u,  228u,  528u, 160u,
            485u,  83u,  382u,  3u,    423u, 248u,  265u,  124u, 124u,
            154u,  288u, 173u,  14u,   4u,   5u,    19u,   303u, 181u,
            13u,   102u, 193u,  470u,  91u,  144u,  21u,   5u,
        };
        EXPECT_VEC_EQ(expected_cherenkov_num_photons,
                      result.cherenkov.num_photons);

        EXPECT_EQ(2101410, result.scintillation.total_num_photons);
        EXPECT_EQ(125, result.scintillation.num_photons.size());
        static unsigned int const expected_scintillation_num_photons[] = {
            27991u, 38157u, 114070u, 114477u, 57893u, 103619u, 90287u, 33901u,
            21827u, 83989u, 55095u,  84026u,  38355u, 3894u,   33219u, 30807u,
            24182u, 41506u, 43246u,  15732u,  28341u, 47956u,  26749u, 47994u,
            22830u, 37627u, 20074u,  38203u,  19233u, 30026u,  17229u, 30547u,
            13618u, 23721u, 4019u,   24306u,  19916u, 19787u,  150u,   19892u,
            17112u, 17217u, 7327u,   17185u,  1110u,  2398u,   25128u, 3256u,
            145u,   20743u, 17817u,  17442u,  7477u,  4858u,   2406u,  4738u,
            143u,   2071u,  206u,    6273u,   12423u, 3695u,   1933u,  255u,
            10312u, 158u,   2214u,   409u,    660u,   18353u,  562u,   9932u,
            14418u, 86u,    155u,    366u,    508u,   2063u,   4340u,  164u,
            25993u, 20u,    1458u,   330u,    21221u, 4758u,   17149u, 152u,
            18048u, 2245u,  800u,    6827u,   526u,   144u,    19621u, 3250u,
            640u,   16117u, 1971u,   164u,    176u,   712u,    993u,   70u,
            1233u,  150u,   882u,    848u,    1599u,  143u,    2670u,  14011u,
            2711u,  144u,   479u,    35243u,  4100u,  157u,    8985u,  4090u,
            21240u, 18420u, 18046u,  9600u,   6725u,
        };
        EXPECT_VEC_EQ(expected_scintillation_num_photons,
                      result.scintillation.num_photons);
    }
    else
    {
        EXPECT_EQ(22335, result.cherenkov.total_num_photons);
        EXPECT_EQ(49, result.cherenkov.num_photons.size());

        EXPECT_SOFT_EQ(
            2101960,
            static_cast<float>(result.scintillation.total_num_photons));

        EXPECT_EQ(117, result.scintillation.num_photons.size());
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
    use_scintillation_ = false;
    auto_flush_ = size_type(-1);
    num_track_slots_ = 4;
    this->build_optical_collector();

    auto result = this->run<MemSpace::host>(4, num_track_slots_, 16);

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
    use_cherenkov_ = false;
    auto_flush_ = size_type(-1);
    num_track_slots_ = 4;
    this->build_optical_collector();

    auto result = this->run<MemSpace::host>(4, num_track_slots_, 16);

    EXPECT_EQ(0, result.cherenkov.total_num_photons);
    EXPECT_EQ(0, result.cherenkov.num_photons.size());

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(1672535, result.scintillation.total_num_photons);
        EXPECT_EQ(48, result.scintillation.num_photons.size());

        // No steps ran
        EXPECT_EQ(0, result.accum.steps);
        EXPECT_EQ(0, result.accum.step_iters);
        EXPECT_EQ(16, result.accum.flushes);
        EXPECT_EQ(0, result.accum.generators.cherenkov);
        EXPECT_EQ(0, result.accum.generators.scintillation);
        EXPECT_EQ(0, result.accum.generators.photons);
    }
    else
    {
        EXPECT_SOFT_EQ(
            1348444,
            static_cast<float>(result.scintillation.total_num_photons));
        EXPECT_EQ(49, result.scintillation.num_photons.size());
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
    auto result = this->run<MemSpace::host>(4, 2, 2);

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(1028, result.accum.steps);
        EXPECT_EQ(33, result.accum.step_iters);
        EXPECT_EQ(1, result.accum.flushes);
        EXPECT_EQ(0, result.accum.generators.cherenkov);
        EXPECT_EQ(2, result.accum.generators.scintillation);
        EXPECT_EQ(1028, result.accum.generators.photons);
    }
}

TEST_F(LArSphereOffloadTest, host_generate)
{
    num_track_slots_ = 262144;
    buffer_capacity_ = 1024;
    initializer_capacity_ = 524288;
    auto_flush_ = 16384;
    this->build_optical_collector();

    // Run with 512 core track slots and 2^18 optical track slots
    auto result = this->run<MemSpace::host>(4, 512, 16);

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(323778, result.accum.steps);
        EXPECT_EQ(2, result.accum.step_iters);
        EXPECT_EQ(1, result.accum.flushes);
        EXPECT_EQ(4, result.accum.generators.cherenkov);
        EXPECT_EQ(4, result.accum.generators.scintillation);
        EXPECT_EQ(323778, result.accum.generators.photons);
    }

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

    auto result = this->run<MemSpace::device>(1, num_track_slots_, 16);
    result.print_expected();

    EXPECT_EQ(7, result.optical_launch_step);
    EXPECT_EQ(0, result.scintillation.total_num_photons);
    EXPECT_EQ(0, result.cherenkov.total_num_photons);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
