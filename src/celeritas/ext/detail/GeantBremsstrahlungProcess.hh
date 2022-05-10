//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantBremsstrahlungProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VEnergyLossProcess.hh>

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
    enum class ModelSelection
    {
        seltzer_berger,
        relativistic,
        all
    };

    // Construct with model selection
    explicit GeantBremsstrahlungProcess(ModelSelection selection);
    // Empty destructor
    ~GeantBremsstrahlungProcess();

    // True for electrons and positrons
    bool IsApplicable(const G4ParticleDefinition& particle) final;
    // Print documentation
    void ProcessDescription(std::ostream&) const override;

  protected:
    // Initialise process by constructing selected models
    void InitialiseEnergyLossProcess(const G4ParticleDefinition*,
                                     const G4ParticleDefinition*) override;
    // Print class parameters
    void StreamProcessInfo(std::ostream& output) const override;

  protected:
    bool is_initialized_;

  private:
    // Hide assignment operator
    GeantBremsstrahlungProcess&
    operator=(const GeantBremsstrahlungProcess& right)
        = delete;
    GeantBremsstrahlungProcess(const GeantBremsstrahlungProcess&) = delete;

  private:
    ModelSelection model_selection_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
