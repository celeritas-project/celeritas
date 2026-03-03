//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarStandaloneRunner.test.cc
//---------------------------------------------------------------------------//
#include "larceler/LarStandaloneRunner.hh"

#include <memory>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>

#include "corecel/io/Repr.hh"
#include "geocel/Types.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "AssertionHelper.hh"
#include "PersistentSP.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"
#include "larceler/Convert.hh"
#include "testdetail/TestMacrosImpl.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class LarStandaloneRunnerTestBase : public ::celeritas::test::Test
{
  protected:
    using Runner = LarStandaloneRunner;
    using Input = inp::OpticalStandaloneInput;
    using VecReal3 = std::vector<Real3>;

    //! Construct input
    virtual Input make_input() const = 0;

    //! Map of larsoft detector ID to actual
    virtual VecReal3 make_detector_point_map() const = 0;

    //! Build runner in the first SetUp of each test suite
    void SetUp() final;

    //! Access the runner
    Runner& runner() const { return *runner_; }

  private:
    std::shared_ptr<Runner> runner_;
};

//---------------------------------------------------------------------------//
void LarStandaloneRunnerTestBase::SetUp()
{
    static PersistentSP<Runner> pr{"LarStandaloneRunner"};

    ::testing::TestInfo const* const test_info
        = ::testing::UnitTest::GetInstance()->current_test_info();
    CELER_ASSERT(test_info);
    std::string test_name{test_info->test_suite_name()};
    pr.lazy_update(test_name, [this]() {
        return std::make_shared<Runner>(this->make_input(),
                                        this->make_detector_point_map());
    });
    runner_ = pr.value();
    CELER_ENSURE(runner_);
}

//---------------------------------------------------------------------------//

struct RunResult
{
    std::vector<int> num_hits;

    static RunResult
    from_btr(std::vector<sim::OpDetBacktrackerRecord> const& records);

    void print_expected() const;
};

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   RunResult const& val1,
                                   RunResult const& val2);

RunResult
RunResult::from_btr(std::vector<sim::OpDetBacktrackerRecord> const& response)
{
    RunResult result;
    for (auto i : range(response.size()))
    {
        // Shouldn't have hits on top detector
        auto const& btr = response[i];
        EXPECT_EQ(i, btr.OpDetNum());
        auto const& hits = btr.timePDclockSDPsMap();
        result.num_hits.push_back(hits.size());
    }
    return result;
}

//---------------------------------------------------------------------------//

void RunResult::print_expected() const
{
    std::cout << "RunResult ref;\n"
              << "ref.num_hits = " << repr(num_hits) << ";\n"
              << "EXPECT_REF_EQ(ref, result);\n";
}

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   RunResult const& val1,
                                   RunResult const& val2)
{
    AssertionHelper helper{expr1, expr2};

    if (auto r = ::celeritas::testdetail::IsVecEq(
            expr1, "num_hits", val1.num_hits, val2.num_hits);
        !static_cast<bool>(r))
    {
        helper.fail() << r.message();
    }

    return helper;
}

//---------------------------------------------------------------------------//

class LarSphereTest : public LarStandaloneRunnerTestBase
{
    //! Construct input
    Input make_input() const override;
    VecReal3 make_detector_point_map() const override;
};

auto LarSphereTest::make_input() const -> Input
{
    Input result;
    result.problem.model.geometry
        = this->test_data_path("geocel", "lar-sphere.gdml");
    result.detectors = {"detshell"};
    result.problem.limits.steps = 16;
    result.problem.capacity = [] {
        inp::OpticalStateCapacity cap;
        cap.tracks = 16;
        cap.primaries = 8 * cap.tracks;
        cap.generators = 512;
        return cap;
    }();
    result.problem.num_streams = 1;
    result.problem.generator = inp::OpticalOffloadGenerator{};
    result.geant_setup.cherenkov = std::nullopt;
    return result;
}

auto LarSphereTest::make_detector_point_map() const -> VecReal3
{
    return {
        from_cm(Real3{0, 105, 0}),
        from_cm(Real3{0, -105, 0}),
    };
}

TEST_F(LarSphereTest, single_sim_edep)
{
    auto& run = this->runner();

    /*
     * See larg4/Services/SimEnergyDepositSD.cc
     * - Number of electrons is arbitrarily set by LArG4
     * - Length unit is cm (LarsoftLen)
     * - Time unit is ns (LarsoftTime)
     * - "original" track ID is always same as actual
     */
    real_type edep{0.1};  // MeV
    LarsoftTime start_time{1.0};
    LarsoftTime end_time{2.0};

    sim::SimEnergyDeposit sed(
        /* numPhotons = */ 32,
        /* numElectrons = */ static_cast<int>(edep * 100),
        /* scintYieldRatio = */ 1.0,
        /* edep = */ edep,
        /* startPos = */ convert_to_larsoft<LarsoftLen>(from_cm(Real3{-1, -98, 0.0})),
        /* endPos = */ convert_to_larsoft<LarsoftLen>(from_cm(Real3{1, -98, 0})),
        /* startTime = */ start_time.value(),
        /* endTime = */ end_time.value(),
        /* trackID = */ 123,
        /* pdgCode = */ pdg::electron().get(),
        /* origTrackID = */ 123);

    auto result = RunResult::from_btr(run({sed}));
    // result.print_expected();
    RunResult ref;
    ref.num_hits = {3, 25};
    EXPECT_REF_EQ(ref, result);

    // Run again (simulating second event)
    result = RunResult::from_btr(run({sed}));
    ref.num_hits = {5, 22};
    EXPECT_REF_EQ(ref, result);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
