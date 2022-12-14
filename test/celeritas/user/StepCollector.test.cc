//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCollector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/user/StepCollector.hh"

#include "corecel/cont/Span.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/global/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "../SimpleTestBase.hh"
#include "../TestEm15Base.hh"
#include "../TestEm3Base.hh"
#include "CaloTestBase.hh"
#include "ExampleCalorimeters.hh"
#include "ExampleMctruth.hh"
#include "MctruthTestBase.hh"
#include "celeritas_test.hh"

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class KnStepCollectorTestBase : public SimpleTestBase,
                                virtual public StepCollectorTestBase
{
  protected:
    VecPrimary make_primaries(size_type count) override
    {
        Primary p;
        p.particle_id = this->particle()->find(pdg::gamma());
        CELER_ASSERT(p.particle_id);
        p.energy    = MevEnergy{10.0};
        p.track_id  = TrackId{0};
        p.position  = {0, 0, 0};
        p.direction = {1, 0, 0};
        p.time      = 0;

        std::vector<Primary> result(count, p);
        for (auto i : range(count))
        {
            result[i].event_id = EventId{i};
        }
        return result;
    }
};

class KnMctruthTest : public KnStepCollectorTestBase, public MctruthTestBase
{
};

class KnCaloTest : public KnStepCollectorTestBase, public CaloTestBase
{
    VecString get_detector_names() const final { return {"inner"}; }
};

//---------------------------------------------------------------------------//

class TestEm3CollectorTestBase : public TestEm3Base,
                                 virtual public StepCollectorTestBase
{
    //! Use MSC
    bool enable_msc() const override { return true; }

    SPConstAction build_along_step() override
    {
        auto&              action_reg = *this->action_reg();
        UniformFieldParams field_params;
        field_params.field = {0, 0, 1 * units::tesla};
        auto result        = AlongStepUniformMscAction::from_params(
            action_reg.next_id(), *this->physics(), field_params);
        CELER_ASSERT(result);
        CELER_ASSERT(result->has_msc() == this->enable_msc());
        action_reg.insert(result);
        return result;
    }

    VecPrimary make_primaries(size_type count) override
    {
        Primary p;
        p.energy    = MevEnergy{10.0};
        p.position  = {-22, 0, 0};
        p.direction = {1, 0, 0};
        p.time      = 0;
        std::vector<Primary> result(count, p);

        auto electron = this->particle()->find(pdg::electron());
        CELER_ASSERT(electron);
        auto positron = this->particle()->find(pdg::positron());
        CELER_ASSERT(positron);

        for (auto i : range(count))
        {
            result[i].event_id    = EventId{0};
            result[i].track_id    = TrackId{i};
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
    VecString get_detector_names() const final
    {
        return {"gap_lv_0", "gap_lv_1", "gap_lv_2"};
    }
};

//---------------------------------------------------------------------------//
// ERROR CHECKING
//---------------------------------------------------------------------------//

TEST_F(KnStepCollectorTestBase, mixing_types)
{
    auto calos = std::make_shared<ExampleCalorimeters>(
        *this->geometry(), std::vector<std::string>{"inner"});
    auto mctruth = std::make_shared<ExampleMctruth>();

    StepCollector::VecInterface interfaces = {calos, mctruth};

    EXPECT_THROW((StepCollector{std::move(interfaces),
                                this->geometry(),
                                this->action_reg().get()}),
                 celeritas::RuntimeError);
}

TEST_F(KnStepCollectorTestBase, multiple_interfaces)
{
    // Add mctruth twice so each step is doubly written
    auto                        mctruth = std::make_shared<ExampleMctruth>();
    StepCollector::VecInterface interfaces = {mctruth, mctruth};
    auto                        collector  = std::make_shared<StepCollector>(
        std::move(interfaces), this->geometry(), this->action_reg().get());

    // Do one step with two tracks
    {
        StepperInput step_inp;
        step_inp.params          = this->core();
        step_inp.num_track_slots = 2;

        Stepper<MemSpace::host> step(step_inp);

        auto primaries = this->make_primaries(2);
        step(make_span(primaries));
    }

    EXPECT_EQ(4, mctruth->steps().size());
}

//---------------------------------------------------------------------------//
// KLEIN-NISHINA
//---------------------------------------------------------------------------//

TEST_F(KnMctruthTest, single_step)
{
    auto result = this->run(8, 1);

    static const int expected_event[] = {0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_VEC_EQ(expected_event, result.event);
    static const int expected_track[] = {0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_EQ(expected_track, result.track);
    static const int expected_step[] = {1, 1, 1, 1, 1, 1, 1, 1};
    EXPECT_VEC_EQ(expected_step, result.step);
    if (!CELERITAS_USE_VECGEOM)
    {
        static const int expected_volume[] = {1, 1, 1, 1, 1, 1, 1, 1};
        EXPECT_VEC_EQ(expected_volume, result.volume);
    }
    static const double expected_pos[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    static const double expected_dir[] = {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                                          1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_dir, result.dir);
}

TEST_F(KnMctruthTest, two_step)
{
    auto result = this->run(4, 2);

    // clang-format off
    static const int expected_event[] = {0, 0, 1, 1, 2, 2, 3, 3};
    EXPECT_VEC_EQ(expected_event, result.event);
    static const int expected_track[] = {0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_EQ(expected_track, result.track);
    static const int expected_step[] = {1, 2, 1, 2, 1, 2, 1, 2};
    EXPECT_VEC_EQ(expected_step, result.step);
    if (!CELERITAS_USE_VECGEOM)
    {
        static const int expected_volume[] = {1, 1, 1, 1, 1, 2, 1, 2};
        EXPECT_VEC_EQ(expected_volume, result.volume);
    }
    static const double expected_pos[] = {0, 0, 0, 2.6999255778482, 0, 0, 0, 0, 0, 3.5717683161497, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    static const double expected_dir[] = {1, 0, 0, 0.45619379667222, 0.14402721708137, -0.87814769863479, 1, 0, 0, 0.8985574206844, -0.27508545475671, -0.34193940152356, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_dir, result.dir);
    // clang-format on
}

TEST_F(KnCaloTest, single_event)
{
    auto result = this->run(1, 64);

    static const double expected_edep[] = {0.00043564799352598};
    EXPECT_VEC_SOFT_EQ(expected_edep, result.edep);
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
        if (!CELERITAS_USE_VECGEOM)
        {
            static const int expected_volume[] = {1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2};
            EXPECT_VEC_EQ(expected_volume, result.volume);
        }
        static const double expected_pos[] = {-22, 0, 0, -20, 0.62729376699828, 0, -19.974881515749, 0.63919659778087, 0.0048217121175519, -19.934025722828, 0.64565957014582, 0.023962145128973, -22, 0, 0, -20, -0.6272937669977, 0, -19.968347491864, -0.64477993221197, 0.0094975131089902, -19.926616737736, -0.65875813304638, 0.027537430238836, -22, 0, 0, -20, 0.62729376699805, 0, -19.972019729151, 0.66426222006667, -0.0037691265086884, -19.982100079658, 0.68574627311443, 0.027941516714081, -22, 0, 0, -20, -0.62729376699778, 0, -19.975083374908, -0.65817577010885, -0.0029322565536058, -19.959498010488, -0.70081357929493, 0.0044325483363873};
        EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
        static const double expected_dir[] = {1, 0, 0, 0.82087264698346, 0.57111128288133, 0, 0.86898307494434, 0.46974361575277, 0.15559354395281, 0.93933735506149, 0.33063938350342, -0.091230101734915, 1, 0, 0, 0.8208726469842, -0.57111128288028, 0, 0.97025863720971, -0.23236953975431, 0.067842272325867, 0.075841950532517, -0.590557814686, 0.80342359067475, 1, 0, 0, 0.82087264698375, 0.57111128288092, 0, -0.21530637869043, 0.77276783694, 0.59705362697691, -0.48956806668328, 0.50481519308475, 0.71097449245019, 1, 0, 0, 0.8208726469841, -0.57111128288041, 0, 0.50418543298446, -0.77669926497628, 0.37753847612074, 0.043343361560974, -0.9667738756985, -0.25193178893401};
        EXPECT_VEC_SOFT_EQ(expected_dir, result.dir);
        // clang-format on
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();
    }
}

TEST_F(TestEm3CaloTest, thirtytwo_step)
{
    auto result = this->run(256, 32);

    static const double expected_edep[]
        = {1535.4185205798, 109.69434829612, 20.443067191226};
    EXPECT_VEC_NEAR(expected_edep, result.edep, 0.5);
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
