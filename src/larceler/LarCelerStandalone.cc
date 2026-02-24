//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarCelerStandalone.cc
//---------------------------------------------------------------------------//
#include "LarCelerStandalone.hh"

#include <memory>
#include <larcore/CoreUtils/ServiceUtil.h>
#include <larcore/Geometry/Geometry.h>
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

    // Obtain the GDML filename from the LAr geometry service
    auto geo_handle = lar::providerFrom<geo::Geometry>();
    CELER_VALIDATE(geo_handle, << "LArSoft geometry is not active");
    result.problem.model.geometry = geo_handle->GDMLFile();

    result.problem.generator = inp::OpticalOffloadGenerator{};

    // Optical capacities
    {
        result.problem.capacity.primaries = 8192;
        result.problem.capacity.tracks = 128;
        result.problem.capacity.generators = 32768;
    }

    result.problem.num_streams = 1;
    result.problem.seed = cfg.seed();
    result.problem.timers.action = cfg.action_times();
    result.problem.output_file = cfg.output_file();

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
