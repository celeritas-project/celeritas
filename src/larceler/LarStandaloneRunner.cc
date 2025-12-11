//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarStandaloneRunner.cc
//---------------------------------------------------------------------------//
#include "LarStandaloneRunner.hh"

#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with input parameters.
 */
LarStandaloneRunner::LarStandaloneRunner(Input const& i)
{
    // For test purposes, create the geometry and immediately tear it down
    auto geo = GeantGeoParams::from_gdml(i.geometry);
    CELER_ASSERT(geo);
    CELER_ASSERT(!global_geant_geo().expired());
}

//---------------------------------------------------------------------------//
/*!
 * Run optical photons from a single set of energy steps.
 */
auto LarStandaloneRunner::operator()(VecSED const&) -> VecBTR
{
    CELER_LOG(error) << "LArSoft interface is incomplete: no hits are "
                        "simulated";
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
