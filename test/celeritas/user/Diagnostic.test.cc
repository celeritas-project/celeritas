//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/Diagnostic.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Span.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/global/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "../GeantTestBase.hh"
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
#define TestEm3ComptonDiagnosticTest \
    TEST_IF_CELERITAS_GEANT(TestEm3ComptonDiagnosticTest)
class TestEm3ComptonDiagnosticTest : public TestEm3Base,
                                     public DiagnosticTestBase
{
  public:
    VecPrimary make_primaries(size_type count) override
    {
        Primary p;
        p.energy = MevEnergy{1.0};
        p.position = {-22, 0, 0};
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

    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = TestEm3Base::build_geant_options();
        opts.compton_scattering = true;
        opts.coulomb_scattering = false;
        opts.photoelectric = false;
        opts.rayleigh_scattering = false;
        opts.gamma_conversion = false;
        opts.gamma_general = false;
        opts.ionization = false;
        opts.annihilation = false;
        opts.brems = BremsModelSelection::none;
        opts.msc = MscModelSelection::none;
        opts.relaxation = RelaxationSelection::none;
        opts.lpm = false;
        opts.eloss_fluctuation = false;
        return opts;
    }
};

//---------------------------------------------------------------------------//
// TESTEM3
//---------------------------------------------------------------------------//

TEST_F(TestEm3DiagnosticTest, host)
{
    auto result = this->run<MemSpace::host>(256, 1024);

    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
        && std::find(result.nonzero_action_keys.begin(),
                     result.nonzero_action_keys.end(),
                     "geo-propagation-limit e+")
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
        static size_type const expected_nonzero_action_counts[] = {124u,
                                                                   391u,
                                                                   476u,
                                                                   19u,
                                                                   59u,
                                                                   1010u,
                                                                   274u,
                                                                   281u,
                                                                   1813u,
                                                                   15u,
                                                                   19u,
                                                                   1186u,
                                                                   1549u,
                                                                   567u,
                                                                   87u,
                                                                   24u,
                                                                   298u};
        EXPECT_VEC_EQ(expected_nonzero_action_counts,
                      result.nonzero_action_counts);

        static size_type const expected_steps[]
            = {0u, 316u, 209u, 91u, 33u, 35u, 21u, 20u, 7u,  11u, 10u,
               1u, 2u,   3u,   3u,  1u,  1u,  1u,  0u,  0u,  1u,  1u,
               0u, 742u, 39u,  11u, 7u,  10u, 6u,  10u, 11u, 5u,  3u,
               7u, 11u,  11u,  10u, 14u, 4u,  4u,  3u,  3u,  3u,  23u,
               0u, 2u,   2u,   1u,  1u,  6u,  5u,  7u,  4u,  6u,  8u,
               7u, 7u,   13u,  8u,  7u,  3u,  2u,  5u,  6u,  2u,  24u};
        EXPECT_VEC_EQ(expected_steps, result.steps);
    }
}

TEST_F(TestEm3DiagnosticTest, TEST_IF_CELER_DEVICE(device))
{
    auto result = this->run<MemSpace::device>(1024, 4);

    // Check action diagnostic results
    static char const* const expected_nonzero_action_keys[]
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
    EXPECT_VEC_EQ(expected_nonzero_action_keys, result.nonzero_action_keys);

    if (this->is_ci_build())
    {
        static size_type const expected_nonzero_action_counts[] = {
            10u, 572u, 508u, 2u, 518u, 521u, 7u, 20u, 21u, 904u, 998u, 12u, 2u, 1u};
        EXPECT_VEC_EQ(expected_nonzero_action_counts,
                      result.nonzero_action_counts);

        static size_type const expected_steps[] = {
            0u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 5u, 2u, 3u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        EXPECT_VEC_EQ(expected_steps, result.steps);

        if (CELERITAS_USE_JSON)
        {
            EXPECT_EQ(
                R"json({"_index":["particle","action"],"actions":[[0,0,0,0,0,0,0,1,0,2,0,0,0,0,0,0,0,0,0,0,7,0],[0,0,0,998,0,0,2,0,0,0,0,21,508,0,0,0,0,0,0,0,521,0],[0,0,0,904,0,0,12,0,0,0,10,20,572,0,0,0,0,0,0,0,518,0]]})json",
                this->action_output());

            EXPECT_EQ(
                R"json({"_index":["particle","num_steps"],"steps":[[0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,5,2,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]})json",
                this->step_output());
        }
    }
}

//---------------------------------------------------------------------------//
// TESTEM3 - Compton scattering only
//---------------------------------------------------------------------------//

TEST_F(TestEm3ComptonDiagnosticTest, host)
{
    auto result = this->run<MemSpace::host>(256, 32);

    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
        && std::find(result.nonzero_action_keys.begin(),
                     result.nonzero_action_keys.end(),
                     "geo-propagation-limit e+")
               != result.nonzero_action_keys.end())
    {
        GTEST_SKIP() << "VecGeom seems to have an edge case where tracks get "
                        "stuck on some builds but not others";
    }

    static char const* const expected_nonzero_action_keys[] = {
        "geo-boundary e-", "geo-boundary gamma", "scat-klein-nishina gamma"};
    EXPECT_VEC_EQ(expected_nonzero_action_keys, result.nonzero_action_keys);

    static size_type const expected_nonzero_action_counts[]
        = {931ul, 6045ul, 1216ul};
    EXPECT_VEC_EQ(expected_nonzero_action_counts, result.nonzero_action_counts);

    static size_type const expected_steps[] = {
        0ul, 0ul, 0ul, 0ul, 8ul, 2ul, 0ul, 0ul,  2ul, 1ul, 4ul, 2ul, 2ul, 3ul,
        1ul, 5ul, 3ul, 1ul, 1ul, 1ul, 3ul, 22ul, 0ul, 0ul, 6ul, 4ul, 3ul, 0ul,
        1ul, 2ul, 2ul, 3ul, 4ul, 1ul, 2ul, 0ul,  0ul, 0ul, 0ul, 0ul, 0ul, 0ul,
        0ul, 1ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul,  0ul, 0ul, 0ul, 0ul, 0ul, 0ul,
        0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul,  0ul, 0ul};
    EXPECT_VEC_EQ(expected_steps, result.steps);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
