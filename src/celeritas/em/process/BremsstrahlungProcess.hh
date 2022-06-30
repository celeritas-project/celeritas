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
    // TODO: update options based on ImportData
    struct Options
    {
        bool combined_model{true};  //!> Use a unified relativistic/SB
                                    //! interactor
        bool enable_lpm{true};      //!> Account for LPM effect at very high
                                    //! energies
        bool use_integral_xs{true}; //!> Use integral method for sampling
                                    //! discrete interaction length
    };

  public:
    // Construct from Bremsstrahlung data
    BremsstrahlungProcess(SPConstParticles particles,
                          SPConstMaterials materials,
                          SPConstImported  process_data,
                          Options          options);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    //! Whether to use the integral method to sample interaction length
    bool use_integral_xs() const final;

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
