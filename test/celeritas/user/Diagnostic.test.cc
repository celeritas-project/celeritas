//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/Diagnostic.test.cc
//---------------------------------------------------------------------------//
#include "corecel/Config.hh"

#include "corecel/cont/Span.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/SimpleTestBase.hh"
#include "celeritas/TestEm3Base.hh"
#include "celeritas/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "DiagnosticTestBase.hh"
#include "TestMacros.hh"
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
    VecPrimary make_primaries(size_type count) const override
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

    if (this->is_ci_build())
    {
        static char const* const expected_nonzero_action_keys[] = {
            "annihil-2-gamma e+",
            "brems-sb e+",
            "brems-sb e-",
            "brems-rel e+",
            "brems-rel e-",
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
            "scat-klein-nishina gamma",
        };
        EXPECT_VEC_EQ(expected_nonzero_action_keys, result.nonzero_action_keys);
        static unsigned int const expected_nonzero_action_counts[] = {
            124u, 402u, 441u, 12u,   66u,   986u, 288u, 286u, 1749u, 16u,
            13u,  18u,  25u,  1195u, 1683u, 534u, 29u,  28u,  297u,
        };
        EXPECT_VEC_EQ(expected_nonzero_action_counts,
                      result.nonzero_action_counts);
        static unsigned int const expected_steps[] = {
            0u, 298u, 226u, 86u, 42u, 27u, 26u, 12u, 8u, 7u, 6u,
            2u, 0u,   3u,   1u,  2u,  3u,  0u,  1u,  0u, 0u, 2u,
            0u, 717u, 40u,  7u,  10u, 14u, 3u,  9u,  8u, 4u, 8u,
            7u, 11u,  9u,   8u,  11u, 5u,  1u,  3u,  4u, 1u, 31u,
            0u, 3u,   1u,   0u,  2u,  4u,  5u,  4u,  6u, 6u, 11u,
            5u, 8u,   7u,   12u, 4u,  6u,  6u,  1u,  2u, 3u, 32u,
        };
        EXPECT_VEC_EQ(expected_steps, result.steps);
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();
    }
}

TEST_F(TestEm3DiagnosticTest, TEST_IF_CELER_DEVICE(device))
{
    auto result = this->run<MemSpace::device>(1024, 4);

    if (this->is_ci_build())
    {
        // Check action diagnostic results
        static char const* const expected_nonzero_action_keys[] = {
            "annihil-2-gamma e+",
            "brems-sb e+",
            "brems-sb e-",
            "geo-boundary e+",
            "geo-boundary e-",
            "geo-boundary gamma",
            "ioni-moller-bhabha e+",
            "ioni-moller-bhabha e-",
            "msc-range e+",
            "msc-range e-",
            "physics-integral-rejected e+",
            "physics-integral-rejected e-",
            "scat-klein-nishina gamma",
        };
        EXPECT_VEC_EQ(expected_nonzero_action_keys, result.nonzero_action_keys);

        static unsigned int const expected_nonzero_action_counts[] = {
            11u, 568u, 509u, 519u, 521u, 10u, 21u, 20u, 908u, 996u, 9u, 2u, 2u};
        EXPECT_VEC_EQ(expected_nonzero_action_counts,
                      result.nonzero_action_counts);

        static unsigned int const expected_steps[] = {
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 5u, 2u, 4u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        };
        EXPECT_VEC_EQ(expected_steps, result.steps);

        EXPECT_JSON_EQ(
            R"json({"_category":"result","_index":["particle","action"],"_label":"action-diagnostic","actions":[[0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,10,0,0,0,0],[0,996,0,0,2,0,0,0,0,20,509,0,0,0,0,0,0,0,0,521,0,0,0,0],[0,908,0,0,9,0,0,0,11,21,568,0,0,0,0,0,0,0,0,519,0,0,0,0]]})json",
            this->action_output());
        EXPECT_JSON_EQ(
            R"json({"_category":"result","_index":["particle","num_steps"],"_label":"step-diagnostic","steps":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,5,2,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]})json",
            this->step_output());
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
