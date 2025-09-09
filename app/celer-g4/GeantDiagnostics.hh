//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/GeantDiagnostics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/io/OutputRegistry.hh"
#include "accel/GeantStepDiagnostic.hh"
#include "accel/SharedParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class MultiExceptionHandler;
class OutputInterface;

namespace app
{
class DetectorConstruction;

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
    using SPConstOutput = std::shared_ptr<OutputInterface const>;
    using SPConstParams = std::shared_ptr<SharedParams const>;
    using SPMultiExceptionHandler = std::shared_ptr<MultiExceptionHandler>;
    using SPOutputRegistry = std::shared_ptr<OutputRegistry>;
    using SPStepDiagnostic = std::shared_ptr<GeantStepDiagnostic>;
    using VecOutputInterface = std::vector<SPConstOutput>;
    //!@}

  public:
    // Add outputs to a queue *only from the main thread*
    static void register_output(VecOutputInterface&&);

    // Construct in an uninitialized state
    GeantDiagnostics() = default;

    // Construct from shared Celeritas params and detectors
    explicit GeantDiagnostics(SharedParams const& params);

    // Initialize diagnostics on the master thread
    inline void Initialize(SharedParams const& params);

    // Write (shared) diagnostic output
    void Finalize();

    // Access the step diagnostic
    inline SPStepDiagnostic const& step_diagnostic() const;

    // Access the exception handler
    inline SPMultiExceptionHandler const& multi_exception_handler() const;

    //! Whether this instance is initialized
    explicit operator bool() const { return static_cast<bool>(meh_); }

  private:
    //// DATA ////

    SPStepDiagnostic step_diagnostic_;
    SPMultiExceptionHandler meh_;

    static VecOutputInterface& queued_output();
};

//---------------------------------------------------------------------------//
/*!
 * Initialize diagnostics on the master thread.
 */
void GeantDiagnostics::Initialize(SharedParams const& params)
{
    *this = GeantDiagnostics(params);
}

//---------------------------------------------------------------------------//
/*!
 * Access the step diagnostic.
 */
auto GeantDiagnostics::step_diagnostic() const -> SPStepDiagnostic const&
{
    CELER_EXPECT(*this);
    return step_diagnostic_;
}

//---------------------------------------------------------------------------//
/*!
 * Access the multi-exception handler.
 */
auto GeantDiagnostics::multi_exception_handler() const
    -> SPMultiExceptionHandler const&
{
    CELER_EXPECT(*this);
    CELER_ENSURE(meh_);
    return meh_;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
