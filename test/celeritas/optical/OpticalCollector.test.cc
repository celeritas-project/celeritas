//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/OpticalCollector.hh"

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
    using BufferId = ItemId<OpticalDistributionData>;
    using BufferRange = ItemRange<OpticalDistributionData>;

    std::shared_ptr<OpticalCollector> collector_;
    StreamId stream_{0};
};

//---------------------------------------------------------------------------//
/*!
 * Construct optical collector at setup time.
 */
void LArSpherePreGenTest::SetUp()
{
    size_type buffer_capacity = 256;
    size_type num_streams = 1;
    auto& action_reg = *this->action_reg();
    collector_ = std::make_shared<OpticalCollector>(this->properties(),
                                                    this->cerenkov(),
                                                    this->scintillation(),
                                                    buffer_capacity,
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
            "result.cerenkov.charge);\n\n"
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

    using ItemsRef
        = Collection<OpticalDistributionData, Ownership::reference, M>;

    auto get_result = [&](PreGenResult& result,
                          ItemsRef const& buffer,
                          size_type size) {
        // Copy buffer to host
        std::vector<OpticalDistributionData> data(size);
        Copier<OpticalDistributionData, MemSpace::host> copy_data{
            make_span(data)};
        copy_data(M, buffer[BufferRange(BufferId(0), BufferId(size))]);

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
    };

    RunResult result;
    auto const& state = collector_->state<M>(stream_);
    auto const& sizes = collector_->num_distributions(stream_);
    get_result(result.cerenkov, state.cerenkov, sizes.cerenkov);
    get_result(result.scintillation, state.scintillation, sizes.scintillation);

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
        = {337u, 503u,  1532u, 1485u, 788u, 610u, 1271u, 433u, 912u, 1051u,
           756u, 1124u, 796u,  854u,  446u, 420u, 582u,  648u, 704u, 825u,
           419u, 496u,  520u,  213u,  338u, 376u, 391u,  517u, 238u, 270u,
           254u, 370u,  23u,   115u,  129u, 317u, 183u,  10u,  1u,   431u,
           301u, 500u,  187u,  373u,  20u,  277u, 145u,  1u};
    EXPECT_VEC_EQ(expected_cerenkov_num_photons, result.cerenkov.num_photons);
    static real_type const expected_cerenkov_charge[] = {-1, 1};
    EXPECT_VEC_EQ(expected_cerenkov_charge, result.cerenkov.charge);

    EXPECT_EQ(106, result.scintillation.num_photons.size());
    static size_type const expected_scintillation_num_photons[]
        = {27991u, 37559u, 114494u, 114637u, 58336u, 45280u, 90827u, 33901u,
           68555u, 74187u, 55095u,  83307u,  53825u, 59271u, 33295u, 30706u,
           42674u, 46522u, 48829u,  59030u,  33903u, 36690u, 38673u, 14395u,
           27169u, 29601u, 30544u,  39639u,  22299u, 23608u, 24698u, 29535u,
           18202u, 19521u, 20364u,  25475u,  10570u, 17164u, 17451u, 21131u,
           187u,   715u,   3100u,   17945u,  720u,   7048u,  13469u, 158u,
           164u,   5993u,  4529u,   167u,    614u,   167u,   450u,   3268u,
           908u,   3872u,  547u,    1188u,   1236u,  418u,   1773u,  2208u,
           5281u,  4127u,  686u,    945u,    6105u,  15114u, 180u,   2960u,
           8114u,  15870u, 1085u,   756u,    157u,   2454u,  699u,   52u,
           152u,   245u,   158u,    1486u,   6033u,  17543u, 3536u,  6809u,
           144u,   4184u,  18u,     157u,    173u,   32953u, 2615u,  618u,
           26619u, 39115u, 21551u,  30985u,  18241u, 24726u, 7999u,  20436u,
           17450u, 3420u};
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
        = {337u, 503u,  1532u, 1485u, 1376u, 1471u, 1153u, 877u, 788u,
           610u, 1271u, 433u,  1068u, 1238u, 110u,  705u,  912u, 1051u,
           756u, 1124u, 779u,  1014u, 594u,  532u,  796u,  854u, 446u,
           420u, 639u,  747u,  354u,  610u,  582u,  648u,  704u, 825u,
           475u, 579u,  827u,  478u,  419u,  496u,  520u,  213u, 107u,
           472u, 712u,  324u,  338u,  376u,  391u,  517u,  6u,   372u,
           675u, 68u,   238u,  270u,  254u,  370u,  315u,  231u, 461u,
           61u,  23u,   115u,  129u,  317u,  188u,  97u,   406u, 183u,
           22u,  268u,  10u,   128u,  26u,   153u,  1u,    105u, 2u};
    EXPECT_VEC_EQ(expected_cerenkov_num_photons, result.cerenkov.num_photons);
    static real_type const expected_cerenkov_charge[] = {-1, 1};
    EXPECT_VEC_EQ(expected_cerenkov_charge, result.cerenkov.charge);

    EXPECT_EQ(193, result.scintillation.num_photons.size());
    static size_type const expected_scintillation_num_photons[]
        = {27991u, 37559u, 114494u, 114637u, 95965u, 107707u, 83200u, 66412u,
           58336u, 45280u, 90827u,  33901u,  82348u, 92196u,  8026u,  50246u,
           68555u, 74187u, 55095u,  83307u,  54711u, 71651u,  42417u, 36917u,
           53825u, 59271u, 33295u,  30706u,  45810u, 56835u,  23107u, 43619u,
           42674u, 46522u, 48829u,  59030u,  36515u, 44766u,  58969u, 34673u,
           33903u, 36690u, 38673u,  14395u,  11257u, 35520u,  54220u, 27632u,
           27169u, 29601u, 30544u,  39639u,  273u,   28493u,  47830u, 5808u,
           22299u, 23608u, 24698u,  29535u,  26582u, 22501u,  37650u, 18666u,
           18202u, 19521u, 20364u,  25475u,  21847u, 19102u,  30255u, 13766u,
           10570u, 17164u, 17451u,  21131u,  18346u, 15207u,  24321u, 715u,
           3100u,  17945u, 9017u,   19929u,  7048u,  7794u,   17347u, 6432u,
           1440u,  11157u, 2595u,   235u,    764u,   1625u,   249u,   158u,
           1814u,  150u,   2792u,   164u,    5993u,  1257u,   5934u,  129u,
           162u,   4529u,  167u,    647u,    1218u,  1983u,   584u,   4625u,
           151u,   6024u,  3268u,   165u,    1800u,  2239u,   5362u,  205u,
           5172u,  240u,   165u,    2991u,   8123u,  7040u,   668u,   947u,
           2064u,  15964u, 12994u,  4721u,   915u,   1076u,   771u,   149u,
           157u,   159u,   7225u,   12628u,  529u,   720u,    642u,   2320u,
           3945u,  8624u,  20u,     2214u,   12011u, 1827u,   144u,   5710u,
           2388u,  2504u,  2096u,   1236u,   12u,    6951u,   1226u,  260u,
           152u,   1496u,  234u,    2739u,   178u,   6033u,   150u,   162u,
           1418u,  1109u,  16u,     836u,    144u,   3751u,   702u,   18319u,
           3663u,  834u,   174u,    5538u,   20758u, 17561u,  9981u,  19808u,
           7358u,  2807u,  164u,    17548u,  4177u,  147u,    16785u, 17370u,
           4272u};
    EXPECT_VEC_EQ(expected_scintillation_num_photons,
                  result.scintillation.num_photons);
    static real_type const expected_scintillation_charge[] = {-1, 0, 1};
    EXPECT_VEC_EQ(expected_scintillation_charge, result.scintillation.charge);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
