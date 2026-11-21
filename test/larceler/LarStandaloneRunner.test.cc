//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarStandaloneRunner.test.cc
//---------------------------------------------------------------------------//
#include "larceler/LarStandaloneRunner.hh"

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <larcoreobj/SimpleTypesAndConstants/geo_vectors.h>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>
#include <nlohmann/json.hpp>

#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/KernelRegistry.hh"
#include "corecel/sys/Version.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/io/OpticalDistributionReader.hh"
#include "celeritas/phys/GeneratorCountersIO.json.hh"  // IWYU pragma: keep
#include "celeritas/phys/PDGNumber.hh"

#include "PersistentSP.hh"
#include "RunnerResults.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{

using VecJson = std::vector<nlohmann::json>;

//---------------------------------------------------------------------------//
/*!
 * Load a newline-delimited JSON file into a vector of JSON objects.
 */
VecJson load_ndjson(std::string const& filename)
{
    CELER_VALIDATE(!filename.empty(), << "no output filename");
    std::ifstream infile(filename);
    CELER_VALIDATE(infile, << "failed to open '" << filename << "'");

    VecJson result;
    std::string line;
    int count{1};
    while (std::getline(infile, line))
    {
        ++count;
        if (line.empty())
        {
            CELER_LOG(warning) << filename << ':' << count << ": empty line";
            result.push_back(nullptr);
        }
        try
        {
            result.push_back(nlohmann::json::parse(line));
        }
        catch (nlohmann::json::exception const&)
        {
            CELER_LOG(critical)
                << filename << ':' << count << ": failed to parse JSON line '"
                << repr(line) << "'";
            throw;
        }
    }
    return result;
}

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
    Runner& runner() const
    {
        CELER_EXPECT(runner_);
        return *runner_;
    }

    // Reset diagnostic output file, returning previous filename
    std::string close_diagnostics();

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
        auto result = std::make_shared<Runner>(
            this->make_input(), this->make_detector_point_map());
        EXPECT_TRUE(result->output_reg().is_open());
        return result;
    });
    runner_ = pr.value();

    CELER_ASSERT(runner_);
    auto& output_reg = const_cast<OutputRegistry&>(runner_->output_reg());
    if (!output_reg.is_open())
    {
        // Probably a subsequent test case
        output_reg.open(this->make_unique_filename(".out.jsonl"));
    }

    CELER_ENSURE(runner_);
}

// Reset diagnostic output file, returning the previous filename (empty if not open)
std::string LarStandaloneRunnerTestBase::close_diagnostics()
{
    CELER_EXPECT(runner_);
    auto& output_reg = const_cast<OutputRegistry&>(runner_->output_reg());
    std::string result;
    if (output_reg.is_open())
    {
        result = output_reg.output_filename();
        output_reg.close();
    }
    return result;
}

//---------------------------------------------------------------------------//

class DuneCryoTest : public LarStandaloneRunnerTestBase
{
  protected:
    //! Construct input
    Input make_input() override;
    VecReal3 make_detector_point_map() const override;

    static std::string& offload_filename()
    {
        static std::string result;
        return result;
    }
};

auto DuneCryoTest::make_input() -> Input
{
    Input result;
    result.problem.output_file = this->make_unique_filename(".out.jsonl");
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
    result.problem.offload_file = this->offload_filename()
        = this->make_unique_filename(".offload.jsonl");
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

TEST_F(DuneCryoTest, setup)
{
    auto diagnostics = load_ndjson(this->close_diagnostics());
    ASSERT_EQ(1, diagnostics.size());
    std::vector<std::string> keys;
    for (auto&& [key, val] : diagnostics.front().items())
    {
        keys.push_back(key);
    }
    static std::string const expected_keys[] = {"input", "internal", "system"};
    EXPECT_VEC_EQ(expected_keys, keys);
}

TEST_F(DuneCryoTest, two_sim_edeps)
{
    auto& run = this->runner();
    bool const is_reference_version
        = Version::from_package("LArSoft") == Version::from_string("10.24.01")
          && Version::from_package("Geant4") == Version::from_string("11.2.2");
    if (!is_reference_version)
    {
        CELER_LOG(info) << "Not using reference versions: "
                        << "LArSoft = " << Version::from_package("LArSoft")
                        << ", Geant4 = " << Version::from_package("Geant4");
    }

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
    RunResult ref{
        /* num_hits = */ {274, 273, 11, 4},
    };

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_REF_EQ(ref, result) << result;
    }

    auto const& sim_channel = raw_result.sim_photons.at(3);
    EXPECT_EQ(3, sim_channel.OpChannel);
    EXPECT_GT(sim_channel.DetectedPhotons.size(), 0);
    // auto hits = raw_result.at(3).TrackIDsAndEnergies(10.0, 20.0); // [ns]

    // Run again (simulating second event)
    result = RunResult::from_btr(run({sed2, sed}).backtrack);
    ref = RunResult{
        /* num_hits = */ {258, 260, 14, 5},
    };
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Run again with varying scintillation yield ratios
    run({make_sed(10, 0.25, 1), make_sed(10, 1.0, 2), make_sed(10, 0.0, 3)});

    // Read distributions written to the offload file
    auto distributions = OpticalDistributionReader(this->offload_filename())();
    EXPECT_EQ(12, distributions.size());

    OpticalDistributionResult optical_result;

    for (auto const& d : distributions)
    {
        optical_result.num_photons.push_back(d.num_photons);
        ASSERT_TRUE(d.component_id);
        optical_result.components.push_back(d.component_id.get());
        ASSERT_TRUE(d.primary);
        optical_result.primaries.push_back(d.primary.get());
    }

    OpticalDistributionResult optical_ref{
        /* num_photons = */ {
            3277, 819, 3277, 819, 3277, 819, 3277, 819, 3, 7, 10, 10},
        /* components = */ {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
        /* primaries = */ {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 2},
    };
    EXPECT_REF_EQ(optical_ref, optical_result) << is_reference_version;

    // Check diagnostic output counters (setup's TearDown should mean this has
    // *only* counters)
    auto diagnostics = load_ndjson(this->close_diagnostics());
    EXPECT_EQ(3, diagnostics.size());

    DiagnosticResult diagnostic_result;

    for (auto& d_json : diagnostics)
    {
        auto stats
            = d_json.at("result").at("counters").get<CounterAccumStats>();
        diagnostic_result.append(stats);
    }

    DiagnosticResult diagnostic_ref{
        /* flushes = */ {1, 1, 1},
        /* num_cut = */ {1429, 1405, 3},
        /* num_errored = */ {0, 0, 0},
        /* step_iters = */ {8, 8, 8},
        /* steps = */ {25393, 25530, 91},
        /* gen_size = */ {1, 1, 1},
        /* buffer_size = */ {4, 4, 4},
        /* num_generated = */ {8192, 8192, 30},
        /* num_pending = */ {8192, 8192, 30},
    };
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_REF_EQ(diagnostic_ref, diagnostic_result) << diagnostic_result;
    }
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

    auto diagnostics = load_ndjson(this->close_diagnostics());
    // NOTE: kernels won't exist if cuda is disabled
    ASSERT_EQ(1, diagnostics.size());
    if (KernelRegistry::profiling())
    {
        EXPECT_JSON_EQ(
            R"json({"result":{"counters":{"flushes":0,"generators":[],"num_cut":0,"num_errored":0,"step_iters":0,"steps":0},"time":{"actions":{},"run":0.0,"setup":0.0,"steps":[],"teardown":0.0}},"system":{"kernels":[]}})json",
            diagnostics.front().dump());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
