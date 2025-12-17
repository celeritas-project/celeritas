//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarCelerStandalone.cc
//---------------------------------------------------------------------------//

#include "LarCelerStandalone.hh"

#include <memory>
#include <art/Utilities/ToolMacros.h>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>

#include "corecel/Assert.hh"

#include "larceler/LarStandaloneRunner.hh"
#include "larceler/inp/LarStandaloneRunner.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with fcl parameters.
 */
LarCelerStandalone::LarCelerStandalone(Parameters const& config)
    : runner_inp_{inp::from_config(config())}
{
}

//---------------------------------------------------------------------------//
/*!
 * Start Celeritas at the beginning of the job.
 */
void LarCelerStandalone::beginJob()
{
    CELER_EXPECT(!runner_);
    runner_ = std::make_unique<LarStandaloneRunner>(runner_inp_);
}

//---------------------------------------------------------------------------//
/*!
 * Run Celeritas on a single event.
 */
auto LarCelerStandalone::executeEvent(VecSED const& edeps) -> UPVecBTR
{
    CELER_EXPECT(runner_);
    CELER_EXPECT(!edeps.empty());

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
}  // namespace celeritas

DEFINE_ART_CLASS_TOOL(celeritas::LarCelerStandalone)
