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
#include "celeritas/user/ActionDiagnostic.hh"

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
        static const size_type expected_action_counts[] = {
            0u,  0u, 0u, 0u, 0u,  0u,   0u,    1580u, 1168u, 0u,   1019u, 55u,
            0u,  0u, 0u, 0u, 22u, 90u,  283u,  0u,    0u,    575u, 0u,    0u,
            19u, 0u, 0u, 0u, 0u,  127u, 0u,    20u,   14u,   0u,   467u,  386u,
            0u,  0u, 0u, 0u, 0u,  0u,   1769u, 289u,  284u,  0u,   15u,   10u,
            0u,  0u, 0u, 0u, 0u,  0u,   0u,    0u,    0u};
        EXPECT_VEC_EQ(expected_action_counts, result.action_counts);

        if (CELERITAS_USE_JSON)
        {
            EXPECT_EQ(
                R"json({"actions":{"annihil-2-gamma e+":127,"brems-combined e+":386,"brems-combined e-":467,"conv-bethe-heitler gamma":19,"eloss-range e+":55,"eloss-range e-":1019,"geo-boundary e+":284,"geo-boundary e-":289,"geo-boundary gamma":1769,"geo-propagation-limit e+":10,"geo-propagation-limit e-":15,"ioni-moller-bhabha e+":14,"ioni-moller-bhabha e-":20,"msc-range e+":1168,"msc-range e-":1580,"photoel-livermore gamma":575,"physics-integral-rejected e+":90,"physics-integral-rejected e-":22,"scat-klein-nishina gamma":283}})json",
                this->action_output());
        }
    }
    else
    {
        // ORANGE results are slightly different
        static const size_type expected_action_counts[] = {
            0u,  0u, 0u, 0u, 0u,  0u,   0u,    1549u, 1186u, 0u,   1010u, 59u,
            0u,  0u, 0u, 0u, 24u, 87u,  298u,  0u,    0u,    567u, 0u,    0u,
            19u, 0u, 0u, 0u, 0u,  124u, 0u,    19u,   15u,   0u,   476u,  391u,
            0u,  0u, 0u, 0u, 0u,  0u,   1813u, 281u,  274u,  0u,   0u,    0u,
            0u,  0u, 0u, 0u, 0u,  0u,   0u,    0u,    0u};
        EXPECT_VEC_EQ(expected_action_counts, result.action_counts);

        if (CELERITAS_USE_JSON)
        {
            EXPECT_EQ(
                R"json({"actions":{"annihil-2-gamma e+":124,"brems-combined e+":391,"brems-combined e-":476,"conv-bethe-heitler gamma":19,"eloss-range e+":59,"eloss-range e-":1010,"geo-boundary e+":274,"geo-boundary e-":281,"geo-boundary gamma":1813,"ioni-moller-bhabha e+":15,"ioni-moller-bhabha e-":19,"msc-range e+":1186,"msc-range e-":1549,"photoel-livermore gamma":567,"physics-integral-rejected e+":87,"physics-integral-rejected e-":24,"scat-klein-nishina gamma":298}})json",
                this->action_output());
        }
    }
}

TEST_F(TestEm3DiagnosticTest, TEST_IF_CELER_DEVICE(device))
{
    auto result = this->run<MemSpace::device>(1024, 4);

    // Check action diagnostic results
    static const size_type expected_action_counts[]
        = {0u, 0u, 0u, 0u, 0u, 0u,  0u, 998u, 904u, 0u, 0u,   0u,
           0u, 0u, 0u, 0u, 2u, 12u, 1u, 0u,   0u,   0u, 0u,   0u,
           2u, 0u, 0u, 0u, 0u, 10u, 0u, 21u,  20u,  0u, 508u, 572u,
           0u, 0u, 0u, 0u, 0u, 0u,  7u, 521u, 518u, 0u, 0u,   0u,
           0u, 0u, 0u, 0u, 0u, 0u,  0u, 0u,   0u};
    EXPECT_VEC_EQ(expected_action_counts, result.action_counts);

    if (CELERITAS_USE_JSON)
    {
        EXPECT_EQ(
            R"json({"actions":{"annihil-2-gamma e+":10,"brems-combined e+":572,"brems-combined e-":508,"conv-bethe-heitler gamma":2,"geo-boundary e+":518,"geo-boundary e-":521,"geo-boundary gamma":7,"ioni-moller-bhabha e+":20,"ioni-moller-bhabha e-":21,"msc-range e+":904,"msc-range e-":998,"physics-integral-rejected e+":12,"physics-integral-rejected e-":2,"scat-klein-nishina gamma":1}})json",
            this->action_output());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
