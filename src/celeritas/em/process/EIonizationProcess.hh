//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/EIonizationProcess.hh
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
 * Ionization process for electrons and positrons.
 */
class EIonizationProcess : public Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using SPConstImported  = std::shared_ptr<const ImportedProcesses>;
    //!@}

    struct EIonizationOptions : Options
    {
        EIonizationOptions() : Options(true) {}
    };

  public:
    // Construct with imported data
    EIonizationProcess(SPConstParticles particles,
                       SPConstImported  process_data);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability applicability) const final;

    //! Get the options for the process
    const Options& options() const final { return options_; }

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles       particles_;
    ImportedProcessAdapter imported_;
    EIonizationOptions     options_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
