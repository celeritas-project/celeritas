//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/BremsstrahlungProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Process.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Bremsstrahlung process for electrons and positrons.
 *
 * Input options are:
 * - \c combined_model: Use a unified model that applies over the entire energy
 *   range of the process
 * - \c enable_lpm: Account for LPM effect at very high energies
 */
class BremsstrahlungProcess : public Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using SPConstMaterials = std::shared_ptr<const MaterialParams>;
    using SPConstImported  = std::shared_ptr<const ImportedProcesses>;
    //!@}

    struct BremsstrahlungOptions : Options
    {
        bool combined_model{true};
        bool enable_lpm{true};

        BremsstrahlungOptions() : Options(true) {}
    };

  public:
    // Construct from Bremsstrahlung data
    BremsstrahlungProcess(SPConstParticles      particles,
                          SPConstMaterials      materials,
                          SPConstImported       process_data,
                          BremsstrahlungOptions options);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    //! Get the options for the process
    const BremsstrahlungOptions& options() const final { return options_; }

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles       particles_;
    SPConstMaterials       materials_;
    ImportedProcessAdapter imported_;
    BremsstrahlungOptions  options_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
