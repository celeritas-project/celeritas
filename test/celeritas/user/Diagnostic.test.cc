//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/Diagnostic.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Span.hh"
#include "corecel/io/StringUtils.hh"
#include "celeritas/UnitUtils.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/global/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "../SimpleTestBase.hh"
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

class SimpleComptonDiagnosticTest : public SimpleTestBase,
                                    public DiagnosticTestBase
{
    VecPrimary make_primaries(size_type count) override
    {
        Primary p;
        p.energy = MevEnergy{10.0};
        p.position = from_cm(Real3{-22, 0, 0});
        p.direction = {1, 0, 0};
        p.time = 0;
        std::vector<Primary> result(count, p);

        auto gamma = this->particle()->find(pdg::gamma());
        CELER_ASSERT(gamma);

        for (auto i : range(count))
        {
            result[i].event_id = EventId{0};
            result[i].track_id = TrackId{i};
            result[i].particle_id = gamma;
        }
        return result;
    }
};

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
            action_reg.next_id(), field_params, nullptr, msc);
        CELER_ASSERT(result);
        CELER_ASSERT(result->has_msc());
        action_reg.insert(result);
        return result;
    }

    VecPrimary make_primaries(size_type count) override
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
            result[i].track_id = TrackId{i};
            result[i].particle_id = (i % 2 == 0 ? electron : positron);
        }
        return result;
    }
};

//---------------------------------------------------------------------------//
// SIMPLE COMPTON
//---------------------------------------------------------------------------//

TEST_F(SimpleComptonDiagnosticTest, host)
{
    auto result = this->run<MemSpace::host>(256, 32);

    static char const* const expected_nonzero_action_keys[]
        = {"geo-boundary electron",
           "geo-boundary gamma",
           "scat-klein-nishina gamma"};
    EXPECT_VEC_EQ(expected_nonzero_action_keys, result.nonzero_action_keys);
    if (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW)
    {
        static size_type const expected_nonzero_action_counts[]
            = {3780u, 525u, 3887u};
        EXPECT_VEC_EQ(expected_nonzero_action_counts,
                      result.nonzero_action_counts);
        static size_type const expected_steps[] = {
            0u, 0u, 0u, 87u, 30u, 10u, 2u, 0u, 1u, 0u,    0u, 3u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u,  0u,  0u,  1u, 0u, 0u, 1840u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u,  0u,  0u,  0u, 0u, 0u, 0u,    0u, 0u, 0u, 0u};
        EXPECT_VEC_EQ(expected_steps, result.steps);
    }
}

//---------------------------------------------------------------------------//
// TESTEM3
//---------------------------------------------------------------------------//

TEST_F(TestEm3DiagnosticTest, host)
{
    auto result = this->run<MemSpace::host>(256, 32);

    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
        && std::find_if(result.nonzero_action_keys.begin(),
                        result.nonzero_action_keys.end(),
                        [](std::string const& s) {
                            return starts_with(s, "geo-propagation-limit");
                        })
               != result.nonzero_action_keys.end())
    {
        GTEST_SKIP() << "VecGeom seems to have an edge case where tracks get "
                        "stuck on some builds but not others";
    }

    // Check action diagnostic results
    static char const* const expected_nonzero_action_keys[]
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
    EXPECT_VEC_EQ(expected_nonzero_action_keys, result.nonzero_action_keys);

    if (this->is_ci_build())
    {
        static size_type const expected_nonzero_action_counts[] = {121ul,
                                                                   389ul,
                                                                   480ul,
                                                                   16ul,
                                                                   57ul,
                                                                   1024ul,
                                                                   276ul,
                                                                   290ul,
                                                                   1798ul,
                                                                   15u,
                                                                   19u,
                                                                   1171ul,
                                                                   1541ul,
                                                                   572ul,
                                                                   86ul,
                                                                   26ul,
                                                                   311ul};

        EXPECT_VEC_EQ(expected_nonzero_action_counts,
                      result.nonzero_action_counts);

        static size_type const expected_steps[]
            = {0u, 308u, 214u, 97u, 42u, 32u, 26u, 17u, 5u,  8u, 8u,
               5u, 2u,   5u,   2u,  0u,  1u,  1u,  1u,  0u,  0u, 1u,
               0u, 756u, 42u,  12u, 10u, 9u,  8u,  5u,  10u, 5u, 3u,
               7u, 10u,  10u,  11u, 13u, 4u,  4u,  4u,  3u,  3u, 23u,
               0u, 2u,   2u,   1u,  2u,  4u,  5u,  7u,  4u,  6u, 7u,
               7u, 7u,   13u,  8u,  6u,  3u,  2u,  5u,  6u,  2u, 24u};
        EXPECT_VEC_EQ(expected_steps, result.steps);
    }
}

TEST_F(TestEm3DiagnosticTest, TEST_IF_CELER_DEVICE(device))
{
    auto result = this->run<MemSpace::device>(1024, 4);

    if (this->is_ci_build())
    {
        // Check action diagnostic results
        static char const* const expected_nonzero_action_keys[]
            = {"annihil-2-gamma e+",
               "brems-combined e+",
               "brems-combined e-",
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
        EXPECT_VEC_EQ(expected_nonzero_action_keys, result.nonzero_action_keys);

        static size_type const expected_nonzero_action_counts[] = {
            10u, 572u, 508u, 518u, 520u, 9u, 20u, 21u, 904u, 997u, 12u, 2u, 3u};
        EXPECT_VEC_EQ(expected_nonzero_action_counts,
                      result.nonzero_action_counts);

        static size_type const expected_steps[] = {
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 5u, 2u, 3u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        EXPECT_VEC_EQ(expected_steps, result.steps);

        if (CELERITAS_USE_JSON)
        {
            EXPECT_EQ(
                "{\"_index\":[\"particle\",\"action\"],\"actions\":[[0,0,0,0,"
                "0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,9,0],[0,0,0,997,0,0,2,0,0,0,"
                "0,21,508,0,0,0,0,0,0,0,520,0],[0,0,0,904,0,0,12,0,0,0,10,20,"
                "572,0,0,0,0,0,0,0,518,0]]}",
                this->action_output());
            EXPECT_EQ(
                "{\"_index\":[\"particle\",\"num_steps\"],\"steps\":[[0,0,0,0,"
                "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,"
                "0,0,0,0,0,0,0,0,0,0,0],[0,0,5,2,3,0,0,0,0,0,0,0,0,0,0,0,0,0,"
                "0,0,0,0]]}",
                this->step_output());
        }
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
