//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarStandaloneRunner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Macros.hh"
#include "celeritas/inp/StandaloneInput.hh"

#include "detail/LarCelerConfig.hh"

namespace sim
{
class SimEnergyDeposit;
class OpDetBacktrackerRecord;
}  // namespace sim

namespace celeritas
{
namespace optical
{
class Runner;
}  // namespace optical

//---------------------------------------------------------------------------//
/*!
 * Setup and run a standalone optical simulation.
 *
 * This class manages the interface between LArSoft data objects and Celeritas.
 * It is separated from the LarCelerStandalone plugin to allow testing
 * and extension to future plugin frameworks (e.g., Phlex).
 * Instantiating the class sets up Celeritas shared and state objects using an
 * input configuration, and each call take a set of energy deposition steps and
 * returns a vector of detector hits.
 *
 * The implementation of this class will set up a standalone celeritas optical
 * simulation using internal Celeritas code to extra hits and "backtracker"
 * data. Conversion between Celeritas objects and the LArSoft data model
 * happens in this class inside the "call" operator.
 *
 * Since LArSoft is single-threaded, this runner uses only a single "stream".
 * We can in theory enable OpenMP to support parallelism across multiple CPUs
 * in a single-process execution.
 *
 * \par Construction
 * See \c celeritas::inp::LarStandaloneRunner .
 */
class LarStandaloneRunner
{
  public:
    //!@{
    //! \name Type aliases
    using VecSED = std::vector<sim::SimEnergyDeposit>;
    using VecBTR = std::vector<sim::OpDetBacktrackerRecord>;
    using Input = inp::OpticalStandaloneInput;
    //!@}

  public:
    // Set up the problem
    explicit LarStandaloneRunner(Input&&);
    // Don't allow copies of this class
    CELER_DEFAULT_MOVE_DELETE_COPY(LarStandaloneRunner);

    // Run optical photons from a single set of energy steps
    VecBTR operator()(VecSED const& edep);

  private:
    using LarsoftTime = Quantity<celeritas::units::Nanosecond, double>;
    using LarsoftLen = Quantity<celeritas::units::Centimeter, double>;

    std::shared_ptr<optical::Runner> runner_;
};

//---------------------------------------------------------------------------//
// Convert from a FHiCL config input
inp::OpticalStandaloneInput
from_config(detail::LarCelerStandaloneConfig const& config);

//---------------------------------------------------------------------------//
}  // namespace celeritas
