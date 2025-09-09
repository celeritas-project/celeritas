//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/MuIonizationProcess.hh
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
 * Ionization process for muons.
 */
class MuIonizationProcess : public Process
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //!@}

    // Options for electron and positron ionization
    struct Options
    {
        //! Maximum energy for the Bragg and ICRU73QO models
        Energy bragg_icru73qo_upper_limit{0.2};  //!< 200 keV
        //! Maximum energy for the Bethe-Bloch model
        Energy bethe_bloch_upper_limit{1e3};  //!< 1 GeV
    };

  public:
    // Construct with imported data
    MuIonizationProcess(SPConstParticles particles,
                        SPConstImported process_data,
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
    ImportedProcessAdapter imported_;
    Options options_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
