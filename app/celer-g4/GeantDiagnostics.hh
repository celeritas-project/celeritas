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

    //! Whether this instance is initialized
    explicit operator bool() const
    {
        return !output_filename_.empty() || !this->any_enabled();
    }

  private:
    //// DATA ////

    std::string output_filename_;
    SPOutputRegistry output_reg_;
    SPStepDiagnostic step_diagnostic_;

    //// HELPER FUNCTIONS ////

    bool any_enabled() const;
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
}  // namespace app
}  // namespace celeritas
