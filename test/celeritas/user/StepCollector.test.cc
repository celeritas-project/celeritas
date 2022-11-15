//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCollector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/user/StepCollector.hh"

#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "../SimpleTestBase.hh"
#include "../TestEm15Base.hh"
#include "../TestEm3Base.hh"
#include "StepCollectorTestBase.hh"
#include "celeritas_test.hh"

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class KnStepCollectorTest : public SimpleTestBase, public StepCollectorTestBase
{
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

//---------------------------------------------------------------------------//

#define TestEm3CollectorTest TEST_IF_CELERITAS_GEANT(TestEm3CollectorTest)
class TestEm3CollectorTest : public TestEm3Base, public StepCollectorTestBase
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
            result[i].event_id    = EventId{i / 2};
            result[i].track_id    = TrackId{i % 2};
            result[i].particle_id = (i % 2 == 0 ? electron : positron);
        }
        return result;
    }
};

//---------------------------------------------------------------------------//
// KLEIN-NISHINA
//---------------------------------------------------------------------------//

TEST_F(KnStepCollectorTest, single_step)
{
    auto result = this->run(8, 1);

    static const int expected_event[] = {0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_VEC_EQ(expected_event, result.event);
    static const int expected_track[] = {0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_EQ(expected_track, result.track);
    static const int expected_step[] = {1, 1, 1, 1, 1, 1, 1, 1};
    EXPECT_VEC_EQ(expected_step, result.step);
    static const int expected_volume[] = {1, 1, 1, 1, 1, 1, 1, 1};
    EXPECT_VEC_EQ(expected_volume, result.volume);
    static const double expected_pos[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    static const double expected_dir[] = {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                                          1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_dir, result.dir);
}

TEST_F(KnStepCollectorTest, two_step)
{
    auto result = this->run(4, 2);

    // clang-format off
    static const int expected_event[] = {0, 0, 1, 1, 2, 2, 3, 3};
    EXPECT_VEC_EQ(expected_event, result.event);
    static const int expected_track[] = {0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_EQ(expected_track, result.track);
    static const int expected_step[] = {1, 2, 1, 2, 1, 2, 1, 2};
    EXPECT_VEC_EQ(expected_step, result.step);
    static const int expected_volume[] = {1, 1, 1, 1, 1, 2, 1, 2};
    EXPECT_VEC_EQ(expected_volume, result.volume);
    static const double expected_pos[] = {0, 0, 0, 2.6999255778482, 0, 0, 0, 0, 0, 3.5717683161497, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    static const double expected_dir[] = {1, 0, 0, 0.45619379667222, 0.14402721708137, -0.87814769863479, 1, 0, 0, 0.8985574206844, -0.27508545475671, -0.34193940152356, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_dir, result.dir);
    // clang-format on
}

//---------------------------------------------------------------------------//
// TESTEM3
//---------------------------------------------------------------------------//

TEST_F(TestEm3CollectorTest, four_step)
{
    auto result = this->run(4, 4);

    // clang-format off
    static const int expected_event[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    EXPECT_VEC_EQ(expected_event, result.event);
    static const int expected_track[] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
    EXPECT_VEC_EQ(expected_track, result.track);
    static const int expected_step[] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    EXPECT_VEC_EQ(expected_step, result.step);
    static const int expected_volume[] = {1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2};
    EXPECT_VEC_EQ(expected_volume, result.volume);
    static const double expected_pos[] = {-22, 0, 0, -20, 0.62729376699826, 0, -19.974880113864, 0.6391963956049, 0.0048227158759443, -19.934033532069, 0.6456599238254, 0.023957201931956, -22, 0, 0, -20, -0.62729376699768, 0, -19.968081209496, -0.64565266924199, 0.0081440702122724, -19.919820127672, -0.66229287124228, 0.030884981104095, -22, 0, 0, -20, 0.62729376699803, 0, -19.972026609161, 0.66425276742328, -0.0037681392058029, -19.982100203076, 0.68573538332315, 0.027933337077879, -22, 0, 0, -20, -0.62729376699775, 0, -19.969797706066, -0.66354021563808, -0.0032805323839768, -19.954139899254, -0.71455552935462, 0.007543633311365};
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    static const double expected_dir[] = {1, 0, 0, 0.82087264698349, 0.5711112828813, 0, 0.86898692949722, 0.46973462662038, 0.1555991546147, 0.93933598017496, 0.33065674679726, -0.091181314676916, 1, 0, 0, 0.82087264698423, -0.57111128288023, 0, 0.97042800149748, -0.23162093552895, 0.067979674420385, -0.049258210356776, -0.57458261870575, 0.81696293856802, 1, 0, 0, 0.82087264698378, 0.57111128288088, 0, -0.21515081877983, 0.77283335382539, 0.59702490098268, -0.48943284066535, 0.50499063855773, 0.71094300013947, 1, 0, 0, 0.82087264698414, -0.57111128288037, 0, 0.4573174367654, -0.78666382478643, 0.41475388943652, -0.062196108539771, -0.95423621311227, -0.29251477512712};
    EXPECT_VEC_SOFT_EQ(expected_dir, result.dir);
    // clang-format on
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
