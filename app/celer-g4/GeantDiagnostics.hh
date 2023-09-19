//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/GeantDiagnostics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/io/OutputRegistry.hh"
#include "accel/GeantStepDiagnostic.hh"
#include "accel/SharedParams.hh"

#include "TimerOutput.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Diagnostics for Geant4 (i.e., for tracks not offloaded to Celeritas).
 *
 * A single instance of this class should be created by the master thread and
 * shared across all threads.
 */
class GeantDiagnostics
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParams = std::shared_ptr<SharedParams const>;
    using SPOutputRegistry = std::shared_ptr<OutputRegistry>;
    using SPStepDiagnostic = std::shared_ptr<GeantStepDiagnostic>;
    using SPTimerOutput = std::shared_ptr<TimerOutput>;
    //!@}

  public:
    // Construct in an uninitialized state
    GeantDiagnostics() = default;

    // Construct from shared Celeritas params on the master thread
    explicit GeantDiagnostics(SPConstParams params);

    // Initialize diagnostics on the master thread
    inline void Initialize(SPConstParams params);

    // Write (shared) diagnostic output
    void Finalize();

    // Access the step diagnostic
    inline SPStepDiagnostic const& StepDiagnostic() const;

    // Access the timer output
    inline SPTimerOutput const& Timer() const;

    //! Whether this instance is initialized
    explicit operator bool() const { return static_cast<bool>(timer_output_); }

  private:
    //// DATA ////

    SPOutputRegistry output_reg_;
    SPStepDiagnostic step_diagnostic_;
    SPTimerOutput timer_output_;
};

//---------------------------------------------------------------------------//
/*!
 * Initialize diagnostics on the master thread.
 */
void GeantDiagnostics::Initialize(SPConstParams params)
{
    *this = GeantDiagnostics(params);
}

//---------------------------------------------------------------------------//
/*!
 * Access the step diagnostic.
 */
auto GeantDiagnostics::StepDiagnostic() const -> SPStepDiagnostic const&
{
    CELER_EXPECT(*this);
    return step_diagnostic_;
}

//---------------------------------------------------------------------------//
/*!
 * Access the timer output.
 */
auto GeantDiagnostics::Timer() const -> SPTimerOutput const&
{
    CELER_EXPECT(*this);
    CELER_EXPECT(timer_output_);
    return timer_output_;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
