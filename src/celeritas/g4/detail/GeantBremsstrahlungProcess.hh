//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/detail/GeantBremsstrahlungProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <G4VEnergyLossProcess.hh>

#include "celeritas/ext/GeantPhysicsOptions.hh"

class G4ParticleDefinition;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Electron/positron Bremsstrahlung process class derived from
 * \c G4eBremsstrahlung . The need for a new process class is to add the option
 * to manually select individual models.
 */
class GeantBremsstrahlungProcess : public G4VEnergyLossProcess
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using ModelSelection = BremsModelSelection;
    //!@}

  public:
    // Construct with model selection and energy limit
    GeantBremsstrahlungProcess(ModelSelection selection,
                               double seltzer_berger_limit);

    // True for electrons and positrons
    bool IsApplicable(G4ParticleDefinition const& particle) final;
    // Print documentation
    void ProcessDescription(std::ostream&) const override;

  protected:
    // Initialise process by constructing selected models
    void InitialiseEnergyLossProcess(G4ParticleDefinition const*,
                                     G4ParticleDefinition const*) override;
    // Print class parameters
    void StreamProcessInfo(std::ostream& output) const override;

  private:
    bool is_initialized_{false};
    ModelSelection model_selection_;
    double sb_limit_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
