//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/Diagnostic.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Span.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/global/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "../TestEm3Base.hh"
#include "DiagnosticTestBase.hh"
#include "celeritas_test.hh"

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

#define TestEm3DiagnosticTest TEST_IF_CELERITAS_GEANT(TestEm3DiagnosticTest)
class TestEm3DiagnosticTest : public TestEm3Base, public DiagnosticTestBase
{
    SPConstAction build_along_step() override
    {
        auto& action_reg = *this->action_reg();
        UniformFieldParams field_params;
        field_params.field = {0, 0, 1 * units::tesla};
        auto msc = UrbanMscParams::from_import(
            *this->particle(), *this->material(), this->imported_data());

        auto result = std::make_shared<AlongStepUniformMscAction>(
            action_reg.next_id(), field_params, msc);
        CELER_ASSERT(result);
        CELER_ASSERT(result->has_msc());
        action_reg.insert(result);
        return result;
    }

    VecPrimary make_primaries(size_type count) override
    {
        Primary p;
        p.energy = MevEnergy{10.0};
        p.position = {-22, 0, 0};
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
            result[i].track_id = TrackId{i};
            result[i].particle_id = (i % 2 == 0 ? electron : positron);
        }
        return result;
    }
};

//---------------------------------------------------------------------------//
// TESTEM3
//---------------------------------------------------------------------------//

TEST_F(TestEm3DiagnosticTest, host)
{
    auto result = this->run<MemSpace::host>(256, 32);

    // Check action diagnostic results
    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM)
    {
        static char const* expected_nonzero_actions[]
            = {"annihil-2-gamma e+",
               "brems-combined e+",
               "brems-combined e-",
               "conv-bethe-heitler gamma",
               "eloss-range e+",
               "eloss-range e-",
               "geo-boundary e+",
               "geo-boundary e-",
               "geo-boundary gamma",
               "geo-propagation-limit e+",
               "geo-propagation-limit e-",
               "ioni-moller-bhabha e+",
               "ioni-moller-bhabha e-",
               "msc-range e+",
               "msc-range e-",
               "photoel-livermore gamma",
               "physics-integral-rejected e+",
               "physics-integral-rejected e-",
               "scat-klein-nishina gamma"};
        EXPECT_VEC_EQ(expected_nonzero_actions, result.nonzero_actions);

        if (this->is_ci_build())
        {
            static const size_type expected_actions[]
                = {0u,   0u,  0u,   0u,    0u,    0u,    0u,  283u, 575u, 19u,
                   0u,   0u,  0u,   0u,    0u,    1769u, 0u,  0u,   0u,   0u,
                   0u,   0u,  0u,   1580u, 1019u, 0u,    22u, 0u,   0u,   0u,
                   0u,   20u, 467u, 0u,    0u,    289u,  15u, 0u,   0u,   0u,
                   0u,   0u,  0u,   1168u, 55u,   0u,    90u, 0u,   0u,   0u,
                   127u, 14u, 386u, 0u,    0u,    284u,  10u, 0u,   0u,   0u};
            EXPECT_VEC_EQ(expected_actions, result.actions);

            static const size_type expected_steps[]
                = {0u, 327u, 196u, 81u, 41u, 29u, 26u, 17u, 13u, 6u, 7u,
                   1u, 3u,   4u,   2u,  1u,  0u,  2u,  1u,  0u,  1u, 1u,
                   0u, 737u, 43u,  17u, 9u,  11u, 7u,  5u,  9u,  5u, 4u,
                   9u, 11u,  9u,   10u, 14u, 4u,  5u,  2u,  2u,  4u, 22u,
                   0u, 0u,   5u,   2u,  1u,  4u,  5u,  8u,  4u,  7u, 7u,
                   7u, 7u,   13u,  8u,  7u,  3u,  2u,  5u,  6u,  1u, 27u};
            EXPECT_VEC_EQ(expected_steps, result.steps);
        }
    }
    else
    {
        // ORANGE results are slightly different
        static char const* expected_nonzero_actions[]
            = {"annihil-2-gamma e+",
               "brems-combined e+",
               "brems-combined e-",
               "conv-bethe-heitler gamma",
               "eloss-range e+",
               "eloss-range e-",
               "geo-boundary e+",
               "geo-boundary e-",
               "geo-boundary gamma",
               "ioni-moller-bhabha e+",
               "ioni-moller-bhabha e-",
               "msc-range e+",
               "msc-range e-",
               "photoel-livermore gamma",
               "physics-integral-rejected e+",
               "physics-integral-rejected e-",
               "scat-klein-nishina gamma"};
        EXPECT_VEC_EQ(expected_nonzero_actions, result.nonzero_actions);

        if (this->is_ci_build())
        {
            static const size_type expected_actions[]
                = {0u,   0u,  0u,   0u,    0u,    0u,    0u,  298u, 567u, 19u,
                   0u,   0u,  0u,   0u,    0u,    1813u, 0u,  0u,   0u,   0u,
                   0u,   0u,  0u,   1549u, 1010u, 0u,    24u, 0u,   0u,   0u,
                   0u,   19u, 476u, 0u,    0u,    281u,  0u,  0u,   0u,   0u,
                   0u,   0u,  0u,   1186u, 59u,   0u,    87u, 0u,   0u,   0u,
                   124u, 15u, 391u, 0u,    0u,    274u,  0u,  0u,   0u,   0u};
            EXPECT_VEC_EQ(expected_actions, result.actions);

            static const size_type expected_steps[]
                = {0u, 316u, 209u, 91u, 33u, 35u, 21u, 20u, 7u,  11u, 10u,
                   1u, 2u,   3u,   3u,  1u,  1u,  1u,  0u,  0u,  1u,  1u,
                   0u, 742u, 39u,  11u, 7u,  10u, 6u,  10u, 11u, 5u,  3u,
                   7u, 11u,  11u,  10u, 14u, 4u,  4u,  3u,  3u,  3u,  23u,
                   0u, 2u,   2u,   1u,  1u,  6u,  5u,  7u,  4u,  6u,  8u,
                   7u, 7u,   13u,  8u,  7u,  3u,  2u,  5u,  6u,  2u,  24u};
            EXPECT_VEC_EQ(expected_steps, result.steps);
        }
    }
}

TEST_F(TestEm3DiagnosticTest, TEST_IF_CELER_DEVICE(device))
{
    auto result = this->run<MemSpace::device>(1024, 4);

    // Check action diagnostic results
    static char const* expected_nonzero_actions[]
        = {"annihil-2-gamma e+",
           "brems-combined e+",
           "brems-combined e-",
           "conv-bethe-heitler gamma",
           "geo-boundary e+",
           "geo-boundary e-",
           "geo-boundary gamma",
           "ioni-moller-bhabha e+",
           "ioni-moller-bhabha e-",
           "msc-range e+",
           "msc-range e-",
           "physics-integral-rejected e+",
           "physics-integral-rejected e-",
           "scat-klein-nishina gamma"};
    EXPECT_VEC_EQ(expected_nonzero_actions, result.nonzero_actions);

    if (this->is_ci_build())
    {
        static const size_type expected_actions[]
            = {0u, 0u, 0u,  0u,  0u,   0u, 0u, 1u,   0u,   2u, 0u,  0u,
               0u, 0u, 0u,  7u,  0u,   0u, 0u, 0u,   0u,   0u, 0u,  998u,
               0u, 0u, 2u,  0u,  0u,   0u, 0u, 21u,  508u, 0u, 0u,  521u,
               0u, 0u, 0u,  0u,  0u,   0u, 0u, 904u, 0u,   0u, 12u, 0u,
               0u, 0u, 10u, 20u, 572u, 0u, 0u, 518u, 0u,   0u, 0u,  0u};
        EXPECT_VEC_EQ(expected_actions, result.actions);

        static const size_type expected_steps[] = {
            0u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 5u, 2u, 3u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        EXPECT_VEC_EQ(expected_steps, result.steps);

        if (CELERITAS_USE_JSON)
        {
            EXPECT_EQ(
                R"json({"_index":["particle","action"],"actions":[[0,0,0,0,0,0,0,1,0,2,0,0,0,0,0,7,0,0,0,0],[0,0,0,998,0,0,2,0,0,0,0,21,508,0,0,521,0,0,0,0],[0,0,0,904,0,0,12,0,0,0,10,20,572,0,0,518,0,0,0,0]]})json",
                this->action_output());

            EXPECT_EQ(
                R"json({"_index":["particle","num_steps"],"steps":[[0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,5,2,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]})json",
                this->step_output());
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
