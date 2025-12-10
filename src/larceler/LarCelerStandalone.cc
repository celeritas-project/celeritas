//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarCelerStandalone.cc
//---------------------------------------------------------------------------//

#include "LarCelerStandalone.hh"

#include <art/Utilities/ToolMacros.h>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>

#include "corecel/Assert.hh"

#include "larceler/LarStandaloneRunner.hh"
#include "larceler/inp/LarStandaloneRunner.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
LarCelerStandalone::LarCelerStandalone(Parameters const& config)
    : runner_inp_{inp::from_config(config())}
{
}

//---------------------------------------------------------------------------//
/*!
 * Instantiate and run Celeritas.
 */
auto LarCelerStandalone::execute(VecSED const& edeps) -> UPVecBTR
{
    CELER_EXPECT(!edeps.empty());

    // Set up GPU, problem, and states
    LarStandaloneRunner run{runner_inp_};

    // Calculate detector responsors for the input steps
    VecBTR result = run(edeps);
    return std::make_unique<VecBTR>(std::move(result));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

DEFINE_ART_CLASS_TOOL(celeritas::LarCelerStandalone)
