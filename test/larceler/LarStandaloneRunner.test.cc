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

#include "geocel/UnitUtils.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/io/OpticalDistributionReader.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "PersistentSP.hh"
#include "RunnerResults.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

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
    virtual Input make_input() = 0;

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

class DuneCryoTest : public LarStandaloneRunnerTestBase
{
  protected:
    //! Construct input
    Input make_input() override;
    VecReal3 make_detector_point_map() const override;
    static std::string& offload_file()
    {
        static std::string result;
        return result;
    }
};

auto DuneCryoTest::make_input() -> Input
{
    Input result;
    result.problem.model.geometry
        = this->test_data_path("geocel", "dune-cryostat.gdml");
    result.detectors = {"PhotonDetector"};
    result.problem.limits.steps = 64;
    result.problem.limits.step_iters = 8;
    result.problem.capacity = [] {
        inp::OpticalStateCapacity cap;
        auto ntracks = 4096_sz;
        cap.tracks = ntracks;
        cap.primaries = 8 * ntracks;
        cap.generators = 512 * ntracks;
        return cap;
    }();
    result.problem.num_streams = 1;
    result.problem.generator = inp::OpticalOffloadGenerator{};
    this->offload_file() = this->make_unique_filename(".jsonl");
    result.problem.offload_file = this->offload_file();
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
    auto make_sed = [](int num_photons, double yield_ratio, int track_id) {
        real_type edep{0.1};  // MeV
        return sim::SimEnergyDeposit(
            /* numPhotons = */ num_photons,
            /* numElectrons = */ static_cast<int>(edep * 100),
            /* scintYieldRatio = */ yield_ratio,
            /* edep = */ edep,
            /* startPos = */ geo::Point_t{5, -712, -540.0},  // [cm]
            /* endPos = */ geo::Point_t{5, -712, -480},  // [cm]
            /* startTime = */ 1.0,  // [ns]
            /* endTime = */ 10.0,  // [ns]
            /* trackID = */ track_id,
            /* pdgCode = */ pdg::electron().get(),
            /* origTrackID = */ track_id);
    };

    auto sed = make_sed(4096, 0.8, 123456789);
    // Make another deposit from energy deposited by a nameless offspring
    // [i.e., no MC truth stored] of Geant4 track ID 1
    sim::SimEnergyDeposit sed2{sed};
    sed2.setTrackID(-1);

    auto raw_result = run({sed, sed2});
    auto result = RunResult::from_btr(raw_result.backtrack);
    RunResult ref;
    ref.num_hits = {274, 273, 11, 4};
    EXPECT_REF_EQ(ref, result);
    // auto hits = raw_result.at(3).TrackIDsAndEnergies(10.0, 20.0); // [ns]

    // Run again (simulating second event)
    result = RunResult::from_btr(run({sed2, sed}).backtrack);
    ref.num_hits = {260, 262, 14, 5};
    EXPECT_REF_EQ(ref, result);

    // Run again with varying scintillation yield ratios
    run({make_sed(10, 0.25, 1), make_sed(10, 1.0, 2), make_sed(10, 0.0, 3)});

    // Read distributions written to the offload file
    auto distributions = OpticalDistributionReader(this->offload_file())();
    EXPECT_EQ(12, distributions.size());

    std::vector<size_type> num_photons;
    std::vector<size_type> components;
    std::vector<size_type> primaries;

    for (auto const& d : distributions)
    {
        num_photons.push_back(d.num_photons);
        ASSERT_TRUE(d.component_id);
        components.push_back(d.component_id.get());
        ASSERT_TRUE(d.primary);
        primaries.push_back(d.primary.get());
    }

    static unsigned int const expected_num_photons[] = {
        3277u, 819u, 3277u, 819u, 3277u, 819u, 3277u, 819u, 3u, 7u, 10u, 10u};
    static unsigned int const expected_components[]
        = {0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u};
    static unsigned int const expected_primaries[]
        = {0u, 0u, 1u, 1u, 0u, 0u, 1u, 1u, 0u, 0u, 1u, 2u};

    EXPECT_VEC_EQ(expected_num_photons, num_photons);
    EXPECT_VEC_EQ(expected_components, components);
    EXPECT_VEC_EQ(expected_primaries, primaries);
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
    EXPECT_TRUE(result.backtrack.empty());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
