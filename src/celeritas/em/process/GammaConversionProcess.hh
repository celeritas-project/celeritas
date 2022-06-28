//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/GammaConversionProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Process.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Conversion of gammas to electrons and positrons.
 *
 * Input options are:
 * - \c enable_lpm: Account for LPM effect at very high energies
 */
class GammaConversionProcess : public Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using SPConstImported  = std::shared_ptr<const ImportedProcesses>;
    //!@}

    struct GammaConversionOptions : Options
    {
        bool enable_lpm{true};

        GammaConversionOptions() : Options(true) {}
    };

  public:
    // Construct from particle data
    GammaConversionProcess(SPConstParticles       particles,
                           SPConstImported        process_data,
                           GammaConversionOptions options);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability applic) const final;

    //! Get the options for the process
    const GammaConversionOptions& options() const final { return options_; }

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles       particles_;
    ImportedProcessAdapter imported_;
    GammaConversionOptions options_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
