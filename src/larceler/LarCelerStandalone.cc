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

namespace celeritas
{
//---------------------------------------------------------------------------//
LarCelerStandalone::LarCelerStandalone(fhicl::ParameterSet const&) {}

//---------------------------------------------------------------------------//
}  // namespace celeritas

DEFINE_ART_CLASS_TOOL(celeritas::LarCelerStandalone)
