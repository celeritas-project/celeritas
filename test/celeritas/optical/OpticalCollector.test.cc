//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/OpticalCollector.hh"

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Copier.hh"
#include "corecel/io/LogContextException.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/global/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/optical/OpticalCollector.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"

#include "celeritas_test.hh"
#include "../LArSphereBase.hh"

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class LArSpherePreGenTest : public LArSphereBase
{
  public:
    using VecPrimary = std::vector<Primary>;

    struct PreGenResult
    {
        std::vector<size_type> num_photons;
        std::vector<real_type> charge;
    };

    struct RunResult
    {
        PreGenResult cerenkov;
        PreGenResult scintillation;

        void print_expected() const;
    };

  public:
    void SetUp() override;

    SPConstAction build_along_step() override;

    VecPrimary make_primaries(size_type count);

    template<MemSpace M>
    RunResult run(size_type num_tracks, size_type num_steps);

  protected:
    using SizeId = ItemId<size_type>;
    using StorageId = ItemId<OpticalDistributionData>;
    using StorageRange = ItemRange<OpticalDistributionData>;

    std::shared_ptr<OpticalCollector> collector_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct optical collector at setup time.
 */
void LArSpherePreGenTest::SetUp()
{
    size_type stack_capacity = 256;
    size_type num_streams = 1;
    auto& action_reg = *this->action_reg();
    collector_ = std::make_shared<OpticalCollector>(this->properties(),
                                                    this->cerenkov(),
                                                    this->scintillation(),
                                                    stack_capacity,
                                                    num_streams,
                                                    &action_reg);
}

//---------------------------------------------------------------------------//
//! Print the expected result
void LArSpherePreGenTest::RunResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "EXPECT_EQ("
         << this->cerenkov.num_photons.size()
         << ", result.cerenkov.num_photons.size());\n"
            "static size_type const expected_cerenkov_num_photons[] = "
         << repr(this->cerenkov.num_photons)
         << ";\n"
            "EXPECT_VEC_EQ(expected_cerenkov_num_photons, "
            "result.cerenkov.num_photons);\n"
            "static real_type const expected_cerenkov_charge[] = "
         << repr(this->cerenkov.charge)
         << ";\n"
            "EXPECT_VEC_EQ(expected_cerenkov_charge, "
            "result.cerenkov.charge);\n"
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
auto LArSpherePreGenTest::build_along_step() -> SPConstAction
{
    auto& action_reg = *this->action_reg();
    UniformFieldParams field_params;
    field_params.field = {0, 0, 1 * units::tesla};
    auto msc = UrbanMscParams::from_import(
        *this->particle(), *this->material(), this->imported_data());

    auto result = std::make_shared<AlongStepUniformMscAction>(
        action_reg.next_id(), field_params, nullptr, msc);
    CELER_ASSERT(result);
    CELER_ASSERT(result->has_msc());
    action_reg.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Generate a vector of primary particles.
 */
auto LArSpherePreGenTest::make_primaries(size_type count) -> VecPrimary
{
    Primary p;
    p.event_id = EventId{0};
    p.energy = MevEnergy{10.0};
    p.position = from_cm(Real3{0, 0, 0});
    p.time = 0;

    Array<ParticleId, 2> const particles = {
        this->particle()->find(pdg::electron()),
        this->particle()->find(pdg::positron()),
    };
    CELER_ASSERT(particles[0] && particles[1]);

    std::vector<Primary> result(count, p);
    IsotropicDistribution<> sample_dir;
    std::mt19937 rng;

    for (auto i : range(count))
    {
        result[i].track_id = TrackId{i};
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
auto LArSpherePreGenTest::run(size_type num_tracks, size_type num_steps)
    -> RunResult
{
    StepperInput step_inp;
    step_inp.params = this->core();
    step_inp.stream_id = StreamId{0};
    step_inp.num_track_slots = num_tracks;

    Stepper<M> step(step_inp);
    LogContextException log_context{this->output_reg().get()};

    // Initial step
    auto primaries = this->make_primaries(num_tracks);
    StepperResult count;
    CELER_TRY_HANDLE(count = step(make_span(primaries)), log_context);

    while (count && --num_steps > 0)
    {
        CELER_TRY_HANDLE(count = step(), log_context);
    }

    using StackRef
        = StackAllocatorData<OpticalDistributionData, Ownership::reference, M>;

    auto get_result = [&](PreGenResult& result, StackRef const& stack) {
        // Copy stack size to host
        std::vector<size_type> size(1);
        Copier<size_type, MemSpace::host> copy_size{make_span(size)};
        copy_size(M, stack.size[AllItems<size_type, M>{}]);

        // Copy stack data to host
        std::vector<OpticalDistributionData> data(size[0]);
        Copier<OpticalDistributionData, MemSpace::host> copy_data{
            make_span(data)};
        copy_data(
            M, stack.storage[StorageRange(StorageId(0), StorageId(size[0]))]);

        std::set<real_type> charge;
        for (auto const& dist : data)
        {
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
        std::sort(std::begin(result.num_photons), std::end(result.num_photons));
    };

    RunResult result;
    auto const& state = collector_->state<M>(StreamId{0});
    get_result(result.cerenkov, state.cerenkov);
    get_result(result.scintillation, state.scintillation);

    return result;
}

//---------------------------------------------------------------------------//
template LArSpherePreGenTest::RunResult
    LArSpherePreGenTest::run<MemSpace::host>(size_type, size_type);
template LArSpherePreGenTest::RunResult
    LArSpherePreGenTest::run<MemSpace::device>(size_type, size_type);

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(LArSpherePreGenTest, host)
{
    auto result = this->run<MemSpace::host>(4, 64);

    EXPECT_EQ(48, result.cerenkov.num_photons.size());
    static size_type const expected_cerenkov_num_photons[]
        = {1u,   1u,   10u,  20u,   23u,   115u,  129u,  145u, 183u, 187u,
           213u, 238u, 254u, 270u,  277u,  301u,  317u,  337u, 338u, 370u,
           373u, 376u, 391u, 419u,  420u,  431u,  433u,  446u, 496u, 500u,
           503u, 517u, 520u, 582u,  610u,  648u,  704u,  756u, 788u, 796u,
           825u, 854u, 912u, 1051u, 1124u, 1271u, 1485u, 1532u};
    EXPECT_VEC_EQ(expected_cerenkov_num_photons, result.cerenkov.num_photons);
    static real_type const expected_cerenkov_charge[] = {-1, 1};
    EXPECT_VEC_EQ(expected_cerenkov_charge, result.cerenkov.charge);

    EXPECT_EQ(106, result.scintillation.num_photons.size());
    static size_type const expected_scintillation_num_photons[]
        = {18u,     52u,    144u,   152u,   157u,   157u,   158u,   158u,
           164u,    167u,   167u,   173u,   180u,   187u,   245u,   418u,
           450u,    547u,   614u,   618u,   686u,   699u,   715u,   720u,
           756u,    908u,   945u,   1085u,  1188u,  1236u,  1486u,  1773u,
           2208u,   2454u,  2615u,  2960u,  3100u,  3268u,  3420u,  3536u,
           3872u,   4127u,  4184u,  4529u,  5281u,  5993u,  6033u,  6105u,
           6809u,   7048u,  7999u,  8114u,  10570u, 13469u, 14395u, 15114u,
           15870u,  17164u, 17450u, 17451u, 17543u, 17945u, 18202u, 18241u,
           19521u,  20364u, 20436u, 21131u, 21551u, 22299u, 23608u, 24698u,
           24726u,  25475u, 26619u, 27169u, 27991u, 29535u, 29601u, 30544u,
           30706u,  30985u, 32953u, 33295u, 33901u, 33903u, 36690u, 37559u,
           38673u,  39115u, 39639u, 42674u, 45280u, 46522u, 48829u, 53825u,
           55095u,  58336u, 59030u, 59271u, 68555u, 74187u, 83307u, 90827u,
           114494u, 114637u};
    EXPECT_VEC_EQ(expected_scintillation_num_photons,
                  result.scintillation.num_photons);
    static real_type const expected_scintillation_charge[] = {-1, 0, 1};
    EXPECT_VEC_EQ(expected_scintillation_charge, result.scintillation.charge);
}

TEST_F(LArSpherePreGenTest, TEST_IF_CELER_DEVICE(device))
{
    auto result = this->run<MemSpace::device>(8, 32);

    EXPECT_EQ(81, result.cerenkov.num_photons.size());
    static size_type const expected_cerenkov_num_photons[]
        = {1u,    2u,    6u,    10u,   22u,   23u,   26u,   61u,   68u,
           97u,   105u,  107u,  110u,  115u,  128u,  129u,  153u,  183u,
           188u,  213u,  231u,  238u,  254u,  268u,  270u,  315u,  317u,
           324u,  337u,  338u,  354u,  370u,  372u,  376u,  391u,  406u,
           419u,  420u,  433u,  446u,  461u,  472u,  475u,  478u,  496u,
           503u,  517u,  520u,  532u,  579u,  582u,  594u,  610u,  610u,
           639u,  648u,  675u,  704u,  705u,  712u,  747u,  756u,  779u,
           788u,  796u,  825u,  827u,  854u,  877u,  912u,  1014u, 1051u,
           1068u, 1124u, 1153u, 1238u, 1271u, 1376u, 1471u, 1485u, 1532u};
    EXPECT_VEC_EQ(expected_cerenkov_num_photons, result.cerenkov.num_photons);
    static real_type const expected_cerenkov_charge[] = {-1, 1};
    EXPECT_VEC_EQ(expected_cerenkov_charge, result.cerenkov.charge);

    EXPECT_EQ(193, result.scintillation.num_photons.size());
    static size_type const expected_scintillation_num_photons[]
        = {12u,    16u,    20u,    129u,   144u,   144u,   147u,    149u,
           150u,   150u,   151u,   152u,   157u,   158u,   159u,    162u,
           162u,   164u,   164u,   165u,   165u,   167u,   174u,    178u,
           205u,   234u,   235u,   240u,   249u,   260u,   273u,    529u,
           584u,   642u,   647u,   668u,   702u,   715u,   720u,    764u,
           771u,   834u,   836u,   915u,   947u,   1076u,  1109u,   1218u,
           1226u,  1236u,  1257u,  1418u,  1440u,  1496u,  1625u,   1800u,
           1814u,  1827u,  1983u,  2064u,  2096u,  2214u,  2239u,   2320u,
           2388u,  2504u,  2595u,  2739u,  2792u,  2807u,  2991u,   3100u,
           3268u,  3663u,  3751u,  3945u,  4177u,  4272u,  4529u,   4625u,
           4721u,  5172u,  5362u,  5538u,  5710u,  5808u,  5934u,   5993u,
           6024u,  6033u,  6432u,  6951u,  7040u,  7048u,  7225u,   7358u,
           7794u,  8026u,  8123u,  8624u,  9017u,  9981u,  10570u,  11157u,
           11257u, 12011u, 12628u, 12994u, 13766u, 14395u, 15207u,  15964u,
           16785u, 17164u, 17347u, 17370u, 17451u, 17548u, 17561u,  17945u,
           18202u, 18319u, 18346u, 18666u, 19102u, 19521u, 19808u,  19929u,
           20364u, 20758u, 21131u, 21847u, 22299u, 22501u, 23107u,  23608u,
           24321u, 24698u, 25475u, 26582u, 27169u, 27632u, 27991u,  28493u,
           29535u, 29601u, 30255u, 30544u, 30706u, 33295u, 33901u,  33903u,
           34673u, 35520u, 36515u, 36690u, 36917u, 37559u, 37650u,  38673u,
           39639u, 42417u, 42674u, 43619u, 44766u, 45280u, 45810u,  46522u,
           47830u, 48829u, 50246u, 53825u, 54220u, 54711u, 55095u,  56835u,
           58336u, 58969u, 59030u, 59271u, 66412u, 68555u, 71651u,  74187u,
           82348u, 83200u, 83307u, 90827u, 92196u, 95965u, 107707u, 114494u,
           114637u};
    EXPECT_VEC_EQ(expected_scintillation_num_photons,
                  result.scintillation.num_photons);
    static real_type const expected_scintillation_charge[] = {-1, 0, 1};
    EXPECT_VEC_EQ(expected_scintillation_charge, result.scintillation.charge);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
