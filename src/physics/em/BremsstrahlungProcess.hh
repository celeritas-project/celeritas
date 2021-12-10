//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BremsstrahlungProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Process.hh"

#include "physics/base/ImportedProcessAdapter.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/material/MaterialParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Options for Bremsstrahlung process
 */
struct BremsstrahlungOptions
{
    bool combined_model{true}; //!> flag for model choices
    bool enable_lpm{true};     //!> flag for the LPM effect
};

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
    using Options          = const BremsstrahlungOptions;
    //!@}

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
