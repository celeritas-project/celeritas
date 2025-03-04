//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCollector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/user/StepCollector.hh"

#include <algorithm>

#include "corecel/cont/Span.hh"
#include "corecel/io/LogContextException.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/user/SimpleCalo.hh"

#include "CaloTestBase.hh"
#include "ExampleInstanceCalo.hh"
#include "ExampleMctruth.hh"
#include "MctruthTestBase.hh"
#include "celeritas_test.hh"
#include "../SimpleTestBase.hh"
#include "../TestEm15Base.hh"
#include "../TestEm3Base.hh"

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class KnSimpleLoopTestBase : public SimpleTestBase,
                             virtual public SimpleLoopTestBase
{
  protected:
    VecPrimary make_primaries(size_type count) const override
    {
        Primary p;
        p.particle_id = this->particle()->find(pdg::gamma());
        CELER_ASSERT(p.particle_id);
        p.energy = MevEnergy{10.0};
        p.position = {0, 0, 0};
        p.direction = {1, 0, 0};
        p.time = 0;

        std::vector<Primary> result(count, p);
        for (auto i : range(count))
        {
            result[i].event_id = EventId{i};
        }
        return result;
    }
};

class KnMctruthTest : public KnSimpleLoopTestBase, public MctruthTestBase
{
};

class KnCaloTest : public KnSimpleLoopTestBase, public CaloTestBase
{
    VecString get_detector_names() const final { return {"inner"}; }
};

//---------------------------------------------------------------------------//

class TestEm3CollectorTestBase : public TestEm3Base,
                                 virtual public SimpleLoopTestBase
{
    SPConstAction build_along_step() override
    {
        auto& action_reg = *this->action_reg();
        UniformFieldParams field_params;
        field_params.field = {0, 0, real_type(1 * units::tesla)};
        auto msc = UrbanMscParams::from_import(
            *this->particle(), *this->material(), this->imported_data());

        auto result = std::make_shared<AlongStepUniformMscAction>(
            action_reg.next_id(), field_params, nullptr, msc);
        CELER_ASSERT(result);
        CELER_ASSERT(result->has_msc());
        action_reg.insert(result);
        return result;
    }

    VecPrimary make_primaries(size_type count) const override
    {
        Primary p;
        p.energy = MevEnergy{10.0};
        p.position = from_cm(Real3{-22, 0, 0});
        p.direction = {1, 0, 0};
        p.time = 0;
        std::vector<Primary> result(count, p);

        auto electron = this->particle()->find(pdg::electron());
        CELER_ASSERT(electron);
        auto positron = this->particle()->find(pdg::positron());
        CELER_ASSERT(positron);

        for (auto i : range(count))
        {
            result[i].event_id = EventId{0};
            result[i].particle_id = (i % 2 == 0 ? electron : positron);
        }
        return result;
    }
};

#define TestEm3MctruthTest TEST_IF_CELERITAS_GEANT(TestEm3MctruthTest)
class TestEm3MctruthTest : public TestEm3CollectorTestBase,
                           public MctruthTestBase
{
};

#define TestEm3CaloTest TEST_IF_CELERITAS_GEANT(TestEm3CaloTest)
class TestEm3CaloTest : public TestEm3CollectorTestBase, public CaloTestBase
{
  public:
    VecString get_detector_names() const final
    {
        return {"gap_0", "gap_1", "gap_2"};
    }
};

#define TestMultiEm3InstanceCaloTest \
    TEST_IF_CELERITAS_GEANT(TestMultiEm3InstanceCaloTest)
class TestMultiEm3InstanceCaloTest : public TestEm3CollectorTestBase
{
  public:
    SPConstAction build_along_step() override
    {
        // Don't use magnetic field
        return TestEm3Base::build_along_step();
    }

    std::string_view geometry_basename() const override
    {
        // NOTE that this is not the flat one, it's the multi-level one.
        return "testem3";
    }

    void SetUp() override
    {
        ExampleInstanceCalo::VecLabel labels = {"lar", "calorimeter", "world"};
        calo_ = std::make_shared<ExampleInstanceCalo>(this->geometry(),
                                                      std::move(labels));
        collector_ = StepCollector::make_and_insert(*this->core(), {calo_});
    }

    template<MemSpace M>
    ExampleInstanceCalo::Result run(size_type num_tracks, size_type num_steps)
    {
        this->run_impl<M>(num_tracks, num_steps);

        CELER_EXPECT(calo_);
        return calo_->result();
    }

  private:
    std::shared_ptr<ExampleInstanceCalo> calo_;
    std::shared_ptr<StepCollector> collector_;
};

//---------------------------------------------------------------------------//
// ERROR CHECKING
//---------------------------------------------------------------------------//

TEST_F(KnSimpleLoopTestBase, mixing_types)
{
    auto calo = std::make_shared<SimpleCalo>(
        std::vector<Label>{"inner"}, *this->geometry(), 1);
    auto mctruth = std::make_shared<ExampleMctruth>();

    StepCollector::VecInterface interfaces = {calo, mctruth};

    EXPECT_THROW((StepCollector{this->geometry(),
                                std::move(interfaces),
                                this->aux_reg().get(),
                                this->action_reg().get()}),
                 celeritas::RuntimeError);
}

TEST_F(KnSimpleLoopTestBase, multiple_interfaces)
{
    // Add mctruth twice so each step is doubly written
    auto mctruth = std::make_shared<ExampleMctruth>();
    auto collector
        = StepCollector::make_and_insert(*this->core(), {mctruth, mctruth});

    // Do one step with two tracks
    {
        StepperInput step_inp;
        step_inp.params = this->core();
        step_inp.stream_id = StreamId{0};
        step_inp.num_track_slots = 2;

        Stepper<MemSpace::host> step(step_inp);

        auto primaries = this->make_primaries(2);
        CELER_TRY_HANDLE(step(make_span(primaries)),
                         LogContextException{this->output_reg().get()});
    }

    EXPECT_EQ(4, mctruth->steps().size());
}

//---------------------------------------------------------------------------//
// KLEIN-NISHINA
//---------------------------------------------------------------------------//

TEST_F(KnMctruthTest, single_step)
{
    auto result = this->run(8, 1);

    static int const expected_event[] = {0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_VEC_EQ(expected_event, result.event);
    static int const expected_track[] = {0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_EQ(expected_track, result.track);
    static int const expected_step[] = {1, 1, 1, 1, 1, 1, 1, 1};
    EXPECT_VEC_EQ(expected_step, result.step);

    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
    {
        static int const expected_volume[] = {1, 1, 1, 1, 1, 1, 1, 1};
        EXPECT_VEC_EQ(expected_volume, result.volume);
    }
    static double const expected_pos[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    static double const expected_dir[] = {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                                          1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_dir, result.dir);
}

TEST_F(KnMctruthTest, two_step)
{
    auto result = this->run(4, 2);

    static int const expected_event[] = {0, 0, 1, 1, 2, 2, 3, 3};
    EXPECT_VEC_EQ(expected_event, result.event);
    static int const expected_track[] = {0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_EQ(expected_track, result.track);
    static int const expected_step[] = {1, 2, 1, 2, 1, 2, 1, 2};
    EXPECT_VEC_EQ(expected_step, result.step);
    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
    {
        static int const expected_volume[] = {1, 1, 1, 1, 1, 2, 1, 2};
        EXPECT_VEC_EQ(expected_volume, result.volume);
    }
    if (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW)
    {
        // clang-format off
        static const double expected_pos[] = {0, 0, 0, 2.6999255778482, 0, 0, 0, 0, 0, 3.5717683161497, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0};
        EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
        static const double expected_dir[] = {1, 0, 0, 0.45619379667222, 0.14402721708137, -0.87814769863479, 1, 0, 0, 0.8985574206844, -0.27508545475671, -0.34193940152356, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
        EXPECT_VEC_SOFT_EQ(expected_dir, result.dir);
        // clang-format on
    }
}

TEST_F(KnCaloTest, single_track)
{
    auto result = this->run<MemSpace::host>(1, 64);

    if (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW)
    {
        static double const expected_edep[] = {0.00043564799352598};
        EXPECT_VEC_SOFT_EQ(expected_edep, result.edep);
    }
    else
    {
        static double const expected_edep[] = {0};
        EXPECT_VEC_SOFT_EQ(expected_edep, result.edep);
    }
}

//---------------------------------------------------------------------------//
// TESTEM3
//---------------------------------------------------------------------------//

TEST_F(TestEm3MctruthTest, four_step)
{
    auto result = this->run(4, 4);

    if (this->is_ci_build() || this->is_summit_build()
        || this->is_wildstyle_build())
    {
        // clang-format off
        static const int expected_event[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        EXPECT_VEC_EQ(expected_event, result.event);
        static const int expected_track[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
        EXPECT_VEC_EQ(expected_track, result.track);
        static const int expected_step[] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
        EXPECT_VEC_EQ(expected_step, result.step);
        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            // 1 is gap_0
            // 101 is world
            static const int expected_volume[] = {101, 1, 1, 1, 101, 1, 1, 1, 101, 1, 1, 1, 101, 1, 1, 1};
            EXPECT_VEC_EQ(expected_volume, result.volume);
        }
        static const double expected_pos[] = {-22, 0, 0, -20, 0.62729376699828,
            0, -19.974880329316, 0.63919631534337, 0.0048226552156834,
            -19.933883384192, 0.64547517416208, 0.02369949278747, -22, 0, 0,
            -20, -0.6272937669977, 0, -19.968081477436, -0.64565253052318,
            0.0081439674481248, -19.912470119913, -0.63911422004052,
            0.011470659775283, -22, 0, 0, -20, 0.62729376699805, 0,
            -19.972026591268, 0.66425280181021, -0.0037681439073629,
            -19.982100207979, 0.68573542038721, 0.027933364396019, -22, 0, 0,
            -20, -0.62729376699778, 0, -19.969797686903, -0.66354024672438,
            -0.0032805361823643, -19.956976222507, -0.71463257305966,
            0.010438618638369};
        EXPECT_VEC_NEAR(expected_pos, result.pos, 1e-11);
        static const double expected_dir[] = {1, 0, 0, 0.82087264698347,
            0.57111128288133, 0, 0.86898688645512, 0.46973495237486,
            0.15559841158064, 0.99921943764862, 0.020164890030908,
            -0.033968995204705, 1, 0, 0, 0.8208726469842, -0.57111128288028, 0,
            0.97042759391972, -0.23162277007504, 0.067979241993019,
            0.98127589918555, 0.041576527794057, 0.18806648296331, 1, 0, 0,
            0.82087264698375, 0.57111128288092, 0, -0.21515133985226,
            0.77283313432545, 0.59702499733971, -0.48943328665481,
            0.50499006011609, 0.71094310398106, 1, 0, 0, 0.8208726469841,
            -0.57111128288041, 0, 0.45731722153487, -0.78666386310603,
            0.41475405407388, 0.35982246270509, -0.81678277174296,
            0.45099190582174};
        EXPECT_VEC_NEAR(expected_dir, result.dir, 1e-10);
        // clang-format on
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated step collector results are required for CI "
                      "tests";
        }
    }
}

TEST_F(TestEm3CaloTest, thirtytwo_step)
{
    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4)
    {
        GTEST_SKIP() << "Track gets stuck with Geant4 navigator";
    }
    auto result = this->run<MemSpace::host>(256, 32);

    static double const expected_edep[]
        = {1548.8862372467, 113.80254412772, 32.259504023678};
    EXPECT_VEC_NEAR(expected_edep, result.edep, 0.5);
}

TEST_F(TestEm3CaloTest, TEST_IF_CELER_DEVICE(step_device))
{
    auto result = this->run<MemSpace::device>(1024, 4);

    static double const expected_edep[] = {1557.5843684091, 0, 0};
    EXPECT_VEC_NEAR(expected_edep, result.edep, 0.5);
}

//---------------------------------------------------------------------------//

TEST_F(TestMultiEm3InstanceCaloTest, step_host)
{
    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
    {
        GTEST_SKIP() << "ORANGE currently does not return physical volume IDs";
    }

    auto result = this->run<MemSpace::host>(128, 256);

    auto iter = std::find(result.instance.begin(),
                          result.instance.end(),
                          "lar:world_PV/Calorimeter/Layer@0.01/lar_pv");
    EXPECT_TRUE(iter != result.instance.end()) << repr(result.instance);
}

TEST_F(TestMultiEm3InstanceCaloTest, TEST_IF_CELER_DEVICE(step_device))
{
    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
    {
        GTEST_SKIP() << "ORANGE currently does not return physical volume IDs";
    }

    auto result = this->run<MemSpace::device>(1024, 32);

    auto iter = std::find(result.instance.begin(),
                          result.instance.end(),
                          "lar:world_PV/Calorimeter/Layer@0.01/lar_pv");
    EXPECT_TRUE(iter != result.instance.end()) << repr(result.instance);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
