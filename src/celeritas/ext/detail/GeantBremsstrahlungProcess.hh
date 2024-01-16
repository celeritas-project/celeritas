//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantBremsstrahlungProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <G4ParticleDefinition.hh>
#include <G4VEnergyLossProcess.hh>

#include "../GeantPhysicsOptions.hh"

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
    using ModelSelection = BremsModelSelection;
    //!@}

  public:
    // Construct with model selection
    explicit GeantBremsstrahlungProcess(ModelSelection selection);

    // True for electrons and positrons
    bool IsApplicable(G4ParticleDefinition const& particle) final;
    // Print documentation
    void ProcessDescription(std::ostream&) const override;

    //! Which models are used
    ModelSelection model_selection() const { return model_selection_; }

  protected:
    // Initialise process by constructing selected models
    void InitialiseEnergyLossProcess(G4ParticleDefinition const*,
                                     G4ParticleDefinition const*) override;
    // Print class parameters
    void StreamProcessInfo(std::ostream& output) const override;

  private:
    bool is_initialized_;
    ModelSelection model_selection_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
