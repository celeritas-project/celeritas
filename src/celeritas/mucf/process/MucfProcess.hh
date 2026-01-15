//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/process/MucfProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"
#include "celeritas/phys/Process.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion of muonic dd, dt, or tt molecules.
 *
 * \note This is an at-rest process.
 */
class MucfProcess final : public Process
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    //!@}

  public:
    // Construct from shared and imported data
    explicit MucfProcess(SPConstParticles particles,
                         SPConstMaterials materials);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    XsGrid macro_xs(Applicability) const final;

    // Get the energy loss for the given energy range
    EnergyLossGrid energy_loss(Applicability) const final;

    //! Whether the integral method can be used to sample interaction length
    bool supports_integral_xs() const final { return false; }

    //! Whether the process applies when the particle is stopped
    bool applies_at_rest() const final { return true; }

    // Name of the process
    std::string_view label() const final;

  private:
    SPConstParticles particles_;
    SPConstMaterials materials_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
