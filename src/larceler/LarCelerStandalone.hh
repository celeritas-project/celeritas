//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarCelerStandalone.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

namespace fhicl
{
class ParameterSet;
}
namespace sim
{
class SimEnergyDeposit;
class OpDetBacktrackerRecord;
}  // namespace sim

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Run optical photons in a standalone simulation.
 */
class LarCelerStandalone
{
  public:
    //!@{
    //! \name Type aliases
    using VecSED = std::vector<sim::SimEnergyDeposit>;
    using UPVecBTR = std::unique_ptr<std::vector<sim::OpDetBacktrackerRecord>>;
    ///@}

    // Construct with fcl parameters
    LarCelerStandalone(fhicl::ParameterSet const& p);

    // Execute simulation
    UPVecBTR execute(VecSED const& edeps);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
