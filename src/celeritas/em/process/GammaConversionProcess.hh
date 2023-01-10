//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/GammaConversionProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/phys/Applicability.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Process.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Conversion of gammas to electrons and positrons.
 */
class GammaConversionProcess : public Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //!@}

    // Options for pair production
    struct Options
    {
        bool enable_lpm{true};  //!< Account for LPM effect at high energies
    };

  public:
    // Construct from particle data
    GammaConversionProcess(SPConstParticles particles,
                           SPConstImported process_data,
                           Options options);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability applic) const final;

    //! Whether to use the integral method to sample interaction length
    bool use_integral_xs() const final { return false; }

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles particles_;
    ImportedProcessAdapter imported_;
    Options options_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
