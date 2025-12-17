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

#include "celeritas/phys/PDGNumber.hh"

#include "PersistentSP.hh"
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
    using Input = LarStandaloneRunner::Input;

    //! Construct input
    virtual Input make_input() const = 0;

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
        return std::make_shared<Runner>(this->make_input());
    });
    runner_ = pr.value();
    CELER_ENSURE(runner_);
}

//---------------------------------------------------------------------------//

class LarSphereTest : public LarStandaloneRunnerTestBase
{
    //! Construct input
    Input make_input() const;
};

auto LarSphereTest::make_input() const -> Input
{
    Input result;
    result.geometry = this->test_data_path("geocel", "lar-sphere.gdml");
    result.optical_step_iters = 10;
    result.optical_capacity.tracks = 16;
    return result;
}

TEST_F(LarSphereTest, single_photon)
{
    auto& run = this->runner();

    // TODO: add helper file with unit conversions
    // - geo::Point_t in cm (larcoreobj/SimpleTypesAndConstants/geo_vectors.h)
    // - time in ns
    real_type edep{0.1};

    /*
     * See larg4/Services/SimEnergyDepositSD.cc
     * - Number of electrons is arbitrarily set by LArG4
     * - Length unit is cm
     * - Time unit is ns
     * - "original" track ID is always same as actual
     */
    sim::SimEnergyDeposit sed(
        /* numPhotons = */ 1,
        /* numElectrons = */ static_cast<int>(edep * 10000),
        /* scintYieldRatio = */ 1.0,
        /* edep = */ edep,
        /* startPos = */ geo::Point_t{0.1, 0.2, 0.3},
        /* endPos = */ geo::Point_t{0.15, 0.24, 0.33},
        /* startTime = */ 1.0,
        /* endTime = */ 2.0,
        /* trackID = */ 123,
        /* pdgCode = */ pdg::electron().get(),
        /* origTrackID = */ 123);

    auto response = run({sed});
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
