//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/BremsstrahlungProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/inp/PhysicsModel.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/Applicability.hh"
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
    using SBInput = inp::SeltzerBergerModel;
    //!@}

    // Options for the Bremsstrahlung process (will be replaced by inp)
    struct Options
    {
        //! Account for LPM effect at very high energies
        bool enable_lpm{true};
    };

  public:
    // Construct from Bremsstrahlung data
    BremsstrahlungProcess(SPConstParticles particles,
                          SPConstMaterials materials,
                          SPConstImported process_data,
                          SBInput sb_input,
                          Options options);

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
    SPConstMaterials materials_;
    ImportedProcessAdapter imported_;
    SBInput sb_input_;
    Options options_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
