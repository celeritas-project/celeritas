//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BremsstrahlungProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "physics/base/ImportedProcessAdapter.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/Process.hh"
#include "physics/material/MaterialParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Bremsstrahlung process for electrons and positrons.
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

    // Options for the Bremsstrahlung process
    struct Options
    {
        bool combined_model{true}; //!> Use a unified relativistic/SB
                                   //! interactor
        bool enable_lpm{true};     //!> Account for LPM effect at very high
                                   //! energies
    };

  public:
    // Construct from Bremsstrahlung data
    BremsstrahlungProcess(SPConstParticles particles,
                          SPConstMaterials materials,
                          SPConstImported  process_data,
                          Options          options);

    // Construct the models associated with this process
    VecModel build_models(ModelIdGenerator next_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    //! Type of process
    ProcessType type() const final;

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles       particles_;
    SPConstMaterials       materials_;
    ImportedProcessAdapter imported_;
    Options                options_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
