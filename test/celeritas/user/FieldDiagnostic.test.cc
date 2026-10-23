//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/FieldDiagnostic.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/user/FieldDiagnostic.hh"

#include <nlohmann/json.hpp>

#include "corecel/io/OutputInterface.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/TestEm3Base.hh"
#include "celeritas/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/inp/Field.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "SimpleLoopTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

#define TestEm3FieldDiagnosticTest \
    TEST_IF_CELERITAS_GEANT(TestEm3FieldDiagnosticTest)
class TestEm3FieldDiagnosticTest : public TestEm3Base, public SimpleLoopTestBase
{
  protected:
    using json = nlohmann::json;

    SPConstAction build_along_step() override
    {
        auto& action_reg = *this->action_reg();

        // Volumes are built with the geometry
        this->geometry();

        // Create the diagnostic before the along-step action that uses it
        inp::FieldDiagnostic diag_inp;
        diag_inp.bins = 20;
        diag_inp.oracle_safety = true;
        diag_inp.step_records = 64;
        diagnostic_ = std::make_shared<FieldDiagnostic>(
            action_reg.next_id(), diag_inp, this->particle(), this->volume(), 1);
        action_reg.insert(diagnostic_);

        UniformFieldParams::Input field_inp;
        field_inp.strength = {0, 0, 1};
        auto msc = UrbanMscParams::from_import(
            *this->particle(), *this->material(), this->imported_data());
        auto result
            = std::make_shared<AlongStepUniformMscAction>(action_reg.next_id(),
                                                          *this->geometry(),
                                                          field_inp,
                                                          nullptr,
                                                          msc,
                                                          diagnostic_);
        action_reg.insert(result);
        return result;
    }

    VecPrimary make_primaries(size_type count) const override
    {
        Primary p;
        p.energy = units::MevEnergy{10.0};
        p.position = from_cm(Real3{-22, 0, 0});
        p.direction = {1, 0, 0};
        p.time = 0;
        std::vector<Primary> result(count, p);

        auto electron = this->particle()->find(pdg::electron());
        auto positron = this->particle()->find(pdg::positron());
        for (auto i : range(count))
        {
            result[i].event_id = EventId{0};
            result[i].particle_id = (i % 2 == 0 ? electron : positron);
        }
        return result;
    }

    // Get the diagnostic output
    json output() const
    {
        CELER_EXPECT(diagnostic_);
        return json::parse(to_string(*diagnostic_));
    }

    // Check that the output is self-consistent
    void check_consistency(json const& out) const;

    std::shared_ptr<FieldDiagnostic> diagnostic_;
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

//! Sum all values in a (possibly nested) JSON array
std::size_t sum(nlohmann::json const& j)
{
    if (j.is_array())
    {
        std::size_t result = 0;
        for (auto const& v : j)
        {
            result += sum(v);
        }
        return result;
    }
    return j.get<std::size_t>();
}

//---------------------------------------------------------------------------//
void TestEm3FieldDiagnosticTest::check_consistency(json const& out) const
{
    auto const& energy = out.at("energy").at("tally");
    auto const& volume = out.at("volume").at("tally");
    auto energy_total
        = [&energy](char const* key) { return sum(energy.at(key)); };

    // Every step is tallied once in each table and histogram
    auto num_steps = energy_total("steps");
    EXPECT_GT(num_steps, 0);
    EXPECT_EQ(num_steps, sum(volume.at("steps")));
    for (auto const& [key, hist] : out.at("step").items())
    {
        SCOPED_TRACE(key);
        EXPECT_EQ(num_steps, sum(hist));
    }
    EXPECT_EQ(num_steps, sum(out.at("step_iter").at(0).at("tracks")));
    EXPECT_GE(num_steps, sum(out.at("safety").at("counts")));

    // The energy and volume tables have the same totals
    for (auto const& [key, values] : energy.items())
    {
        SCOPED_TRACE(key);
        EXPECT_EQ(sum(values), sum(volume.at(key)));
    }

    // One boundary search and outcome per substep
    auto num_substeps = energy_total("substeps");
    EXPECT_GE(num_substeps, num_steps);
    EXPECT_EQ(num_substeps, energy_total("intersects"));
    EXPECT_EQ(num_substeps,
              energy_total("accepted") + energy_total("bumped")
                  + energy_total("intercepted") + energy_total("shortened"));
    EXPECT_EQ(energy_total("integrations"),
              energy_total("quick_advances") + energy_total("chord_iterations")
                  + energy_total("accurate_iterations")
                  + energy_total("good_step_rejects"));
    EXPECT_LE(energy_total("skippable_intersects"),
              energy_total("intersects") - energy_total("intersect_hits"));
    EXPECT_GT(energy_total("boundary"), 0);
    EXPECT_EQ(0, energy_total("safeties"));

    // Propagation occurred in several calorimeter volumes
    EXPECT_GT(out.at("volume").at("label").size(), 2);
    EXPECT_EQ(out.at("volume").at("label").size(), volume.at("steps").size());
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TestEm3FieldDiagnosticTest, host)
{
    this->run_impl<MemSpace::host>(256, 32);
    auto out = this->output();
    this->check_consistency(out);

    // Only charged particles are propagated in the field
    auto const& substeps = out.at("step").at("substeps");
    auto gamma = this->particle()->find(pdg::gamma());
    EXPECT_EQ(0, sum(substeps.at(gamma.get())));
    auto electron = this->particle()->find(pdg::electron());
    EXPECT_GT(sum(substeps.at(electron.get())), 0);

    // The step iteration records match the number of steps taken
    auto const& iters = out.at("step_iter").at(0);
    EXPECT_EQ(32, iters.at("tracks").size());
    EXPECT_FALSE(iters.contains("_truncated"));
}

TEST_F(TestEm3FieldDiagnosticTest, TEST_IF_CELER_DEVICE(device))
{
    this->run_impl<MemSpace::device>(1024, 4);
    auto out = this->output();
    this->check_consistency(out);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
