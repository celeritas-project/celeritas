//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/BremsstrahlungProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>

#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/io/ImportSBTable.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/Applicability.hh"
#include "celeritas/phys/AtomicNumber.hh"
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
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    using ReadData = std::function<ImportSBTable(AtomicNumber)>;
    //!@}

    // Options for the Bremsstrahlung process
    // TODO: update options based on ImportData
    struct Options
    {
        BremsModelSelection selection{BremsModelSelection::all};  //!> Model
                                                                  //! selection
        bool combined_model{false};  //!> Use a unified relativistic/SB
                                     //! interactor
        bool enable_lpm{true};  //!> Account for LPM effect at very high
                                //! energies
        bool use_integral_xs{true};  //!> Use integral method for sampling
                                     //! discrete interaction length
    };

  public:
    // Construct from Bremsstrahlung data
    BremsstrahlungProcess(SPConstParticles particles,
                          SPConstMaterials materials,
                          SPConstImported process_data,
                          ReadData load_sb_table,
                          Options options);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    //! Whether to use the integral method to sample interaction length
    bool use_integral_xs() const final { return options_.use_integral_xs; }

    // Name of the process
    std::string_view label() const final;

  private:
    SPConstParticles particles_;
    SPConstMaterials materials_;
    ImportedProcessAdapter imported_;
    ReadData load_sb_;
    Options options_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
