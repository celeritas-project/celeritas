//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarStandaloneRunner.test.cc
//---------------------------------------------------------------------------//
#include "larceler/LarStandaloneRunner.hh"

#include <fstream>
#include <memory>
#include <larcoreobj/SimpleTypesAndConstants/geo_vectors.h>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>
#include <nlohmann/json.hpp>

#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/KernelRegistry.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/io/OpticalDistributionReader.hh"
#include "celeritas/phys/GeneratorCountersIO.json.hh"
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

    auto const& sim_channel = raw_result.sim_photons.at(3);
    EXPECT_EQ(3, sim_channel.OpChannel);
    EXPECT_GT(sim_channel.DetectedPhotons.size(), 0);
    // auto hits = raw_result.at(3).TrackIDsAndEnergies(10.0, 20.0); // [ns]

    // Run again (simulating second event)
    result = RunResult::from_btr(run({sed2, sed}).backtrack);
    ref.num_hits = {260, 262, 14, 5};
    EXPECT_REF_EQ(ref, result);

    // Run again with varying scintillation yield ratios
    run({make_sed(10, 0.25, 1), make_sed(10, 1.0, 2), make_sed(10, 0.0, 3)});

    // Read distributions written to the offload file
    auto distributions = OpticalDistributionReader(this->offload_filename())();
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

    // Check diagnostic output counters (setup's TearDown should mean this has
    // *only* counters)
    auto diagnostics = load_ndjson(this->close_diagnostics());
    EXPECT_EQ(3, diagnostics.size());

    std::vector<size_type> flushes;
    std::vector<size_type> num_cut;
    std::vector<size_type> num_errored;
    std::vector<size_type> step_iters;
    std::vector<size_type> steps;
    std::vector<size_type> gen_size;
    std::vector<size_type> buffer_size;
    std::vector<size_type> num_generated;
    std::vector<size_type> num_pending;

    for (auto& d_json : diagnostics)
    {
        auto stats
            = d_json.at("result").at("counters").get<CounterAccumStats>();
        flushes.push_back(stats.flushes);
        num_cut.push_back(stats.num_cut);
        num_errored.push_back(stats.num_errored);
        step_iters.push_back(stats.step_iters);
        steps.push_back(stats.steps);
        gen_size.push_back(stats.generators.size());
        for (auto& g : stats.generators)
        {
            buffer_size.push_back(g.buffer_size);
            num_generated.push_back(g.num_generated);
            num_pending.push_back(g.num_pending);
        }
    }

    static unsigned int const expected_flushes[] = {1u, 2u, 3u};
    static unsigned int const expected_num_cut[] = {1696u, 3378u, 3381u};
    static unsigned int const expected_num_errored[] = {0u, 0u, 0u};
    static unsigned int const expected_step_iters[] = {8u, 16u, 24u};
    static unsigned int const expected_steps[] = {25393u, 50923u, 51014u};
    static unsigned int const expected_gen_size[] = {1u, 1u, 1u};
    static unsigned int const expected_buffer_size[] = {4u, 4u, 4u};
    static unsigned int const expected_num_generated[] = {8192u, 8192u, 30u};
    static unsigned int const expected_num_pending[] = {0u, 0u, 0u};

    EXPECT_VEC_EQ(expected_flushes, flushes);
    EXPECT_VEC_EQ(expected_num_cut, num_cut);
    EXPECT_VEC_EQ(expected_num_errored, num_errored);
    EXPECT_VEC_EQ(expected_step_iters, step_iters);
    EXPECT_VEC_EQ(expected_steps, steps);
    EXPECT_VEC_EQ(expected_gen_size, gen_size);
    EXPECT_VEC_EQ(expected_buffer_size, buffer_size);
    EXPECT_VEC_EQ(expected_num_generated, num_generated);
    EXPECT_VEC_EQ(expected_num_pending, num_pending);
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
