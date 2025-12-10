//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarCelerStandalone.cc
//---------------------------------------------------------------------------//

#include "LarCelerStandalone.hh"

#include <art/Utilities/ToolMacros.h>
#include <fhiclcpp/ParameterSet.h>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
LarCelerStandalone::LarCelerStandalone(fhicl::ParameterSet const&) {}

//---------------------------------------------------------------------------//
auto LarCelerStandalone::execute(VecSED const& edeps) -> UPVecBTR
{
    CELER_EXPECT(!edeps.empty());

    CELER_NOT_IMPLEMENTED("LarCelerStandalone");

    VecBTR result;
    // TODO: result from standalone execution
    return std::make_unique<VecBTR>(std::move(result));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

DEFINE_ART_CLASS_TOOL(celeritas::LarCelerStandalone)
