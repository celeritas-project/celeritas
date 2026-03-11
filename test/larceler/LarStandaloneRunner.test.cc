//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarStandaloneRunner.test.cc
//---------------------------------------------------------------------------//
#include "larceler/LarStandaloneRunner.hh"

#include <memory>
#include <larcoreobj/SimpleTypesAndConstants/geo_vectors.h>
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

class DuneCryoTest : public LarStandaloneRunnerTestBase
{
    //! Construct input
    Input make_input() const override;
    VecReal3 make_detector_point_map() const override;
};

auto DuneCryoTest::make_input() const -> Input
{
    Input result;
    result.problem.model.geometry
        = this->test_data_path("geocel", "dune-cryostat.gdml");
    result.detectors = {"PhotonDetector"};
    result.problem.limits.steps = 64;
    result.problem.limits.step_iters = 8;
    result.problem.capacity = [] {
        inp::OpticalStateCapacity cap;
        cap.tracks = 4096;
        cap.primaries = 8 * cap.tracks;
        cap.generators = 512 * cap.tracks;
        return cap;
    }();
    result.problem.num_streams = 1;
    result.problem.generator = inp::OpticalOffloadGenerator{};
    result.geant_setup.cherenkov = std::nullopt;
    return result;
}

auto DuneCryoTest::make_detector_point_map() const -> VecReal3
{
    return {
        from_cm(Real3{-0.05, -712.31875, -535.175}),
        from_cm(Real3{-0.05, -712.31875, -486.375}),
        from_cm(Real3{-0.05, -712.31875, -423.575}),
        from_cm(Real3{-0.05, -712.31875, -374.775}),
    };
}

TEST_F(DuneCryoTest, two_sim_edeps)
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
    sim::SimEnergyDeposit sed(
        /* numPhotons = */ 4096,
        /* numElectrons = */ static_cast<int>(edep * 100),
        /* scintYieldRatio = */ 1.0,
        /* edep = */ 0.1,  // [MeV]
        /* startPos = */ geo::Point_t{5, -712, -540.0},  // [cm]
        /* endPos = */ geo::Point_t{5, -712, -480},  // [cm]
        /* startTime = */ 1.0,  // [ns]
        /* endTime = */ 10.0,  // [ns]
        /* trackID = */ 123456789,
        /* pdgCode = */ pdg::electron().get(),
        /* origTrackID = */ 123456789);
    // Make another deposit from energy deposited by a nameless offspring
    // [i.e., no MC truth stored] of Geant4 track ID 1
    sim::SimEnergyDeposit sed2{sed};
    sed2.setTrackID(-1);

    auto raw_result = run({sed, sed2});
    auto result = RunResult::from_btr(raw_result);
    RunResult ref;
    ref.num_hits = {255, 250, 16, 6};
    EXPECT_REF_EQ(ref, result);
    // auto hits = raw_result.at(3).TrackIDsAndEnergies(10.0, 20.0); // [ns]

    // Run again (simulating second event)
    result = RunResult::from_btr(run({sed2, sed}));
    ref.num_hits = {252, 273, 13, 6};
    EXPECT_REF_EQ(ref, result);
}

TEST_F(DuneCryoTest, zero_photons)
{
    auto& run = this->runner();

    sim::SimEnergyDeposit sed(
        /* numPhotons = */ 0,
        /* numElectrons = */ 100,
        /* scintYieldRatio = */ 1.0,
        /* edep = */ 0.1,
        /* startPos = */ geo::Point_t{-1, -98, 0.0},  // [cm]
        /* endPos = */ geo::Point_t{1, -98, 0},  // [cm]
        /* startTime = */ 1.0,
        /* endTime = */ 1.1,
        /* trackID = */ 123456789,
        /* pdgCode = */ pdg::electron().get(),
        /* origTrackID = */ 123);

    // No run should occur, BTRs should be empty
    auto result = run({sed});
    EXPECT_TRUE(result.empty());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
