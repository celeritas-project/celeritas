//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/MuPairProductionProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/inp/PhysicsModel.hh"
#include "celeritas/phys/Applicability.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Process.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Electron-positron pair production process for muons.
 */
class MuPairProductionProcess : public Process
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    using ModelInput = inp::MuPairProductionModel;
    //!@}

  public:
    // Construct from pair production data
    MuPairProductionProcess(SPConstParticles particles,
                            SPConstImported process_data,
                            ModelInput input);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    XsGrid macro_xs(Applicability range) const final;

    // Get the energy loss for the given energy range
    EnergyLossGrid energy_loss(Applicability range) const final;

    //! Whether the integral method can be used to sample interaction length
    bool supports_integral_xs() const final { return true; }

    //! Whether the process applies when the particle is stopped
    bool applies_at_rest() const final { return imported_.applies_at_rest(); }

    // Name of the process
    std::string_view label() const final;

  private:
    SPConstParticles particles_;
    ImportedProcessAdapter imported_;
    ModelInput table_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
