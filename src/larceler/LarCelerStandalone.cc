//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarCelerStandalone.cc
//---------------------------------------------------------------------------//
#include "LarCelerStandalone.hh"

#include <memory>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>

#include "corecel/Assert.hh"
#include "celeritas/inp/StandaloneInput.hh"

#include "LarStandaloneRunner.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Convert from a FHiCL config input.
 */
inp::OpticalStandaloneInput
make_input_from_config(detail::LarCelerStandaloneConfig const& cfg)
{
    inp::OpticalStandaloneInput result;

#if 0
    // FIXME: environment config doesn't yet work
    {
        fhicl::ParameterSet const& ps = cfg.environment();
        for (auto const& key : ps.get_names()) {
            result.environment[key] = ps.get<std::string>(key);
        }
    }
#endif

    result.problem.model.geometry = cfg.geometry();
    result.problem.generator = inp::OpticalOffloadGenerator{};

    // Optical capacities
    {
        result.problem.capacity.primaries = 8192;
        result.problem.capacity.tracks = 128;
        result.problem.capacity.generators = 32768;
    }

    result.problem.num_streams = 1;
    result.problem.timers.action = true;
    result.problem.output_file = "celer.out.json";

    return result;
}
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with fcl parameters.
 */
LarCelerStandalone::LarCelerStandalone(Parameters const& config)
    : runner_inp_{make_input_from_config(config())}
{
}

//---------------------------------------------------------------------------//
/*!
 * Start Celeritas at the beginning of the job.
 */
void LarCelerStandalone::beginJob()
{
    CELER_EXPECT(!runner_);
    runner_ = std::make_unique<LarStandaloneRunner>(
        std::forward<LarStandaloneRunner::Input>(runner_inp_));
}

//---------------------------------------------------------------------------//
/*!
 * Run Celeritas on a single event.
 */
auto LarCelerStandalone::executeEvent(VecSED const& edeps) -> UPVecBTR
{
    CELER_EXPECT(runner_);
    CELER_EXPECT(!edeps.empty());

    using VecBTR = LarStandaloneRunner::VecBTR;

    // Calculate detector responsors for the input steps
    auto& run = *runner_;
    VecBTR result = run(edeps);
    return std::make_unique<VecBTR>(std::move(result));
}

//---------------------------------------------------------------------------//
/*!
 * Free Celeritas memory at the end of the job.
 */
void LarCelerStandalone::endJob()
{
    CELER_EXPECT(runner_);
    runner_.reset();
}

//---------------------------------------------------------------------------//
/*!
 * No RNG initialization is needed.
 */
void LarCelerStandalone::InitializeTools(CLHEP::HepRandomEngine&,
                                         CLHEP::HepRandomEngine&)
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
