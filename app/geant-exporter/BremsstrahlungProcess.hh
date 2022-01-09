//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BremsstrahlungProcess.hh
//! \brief Bremsstrahlung process class.
//---------------------------------------------------------------------------//
#pragma once

#include <G4VEnergyLossProcess.hh>

namespace geant_exporter
{
//---------------------------------------------------------------------------//
/*!
 * Electron/positron Bremsstrahlung process class derived from
 * \c G4eBremsstrahlung . The need for a new process class is to add the option
 * to manually select individual models.
 */
class BremsstrahlungProcess : public G4VEnergyLossProcess
{
  public:
    enum class ModelSelection
    {
        seltzer_berger,
        relativistic,
        all
    };

    // Construct with model selection
    explicit BremsstrahlungProcess(ModelSelection selection);
    // Empty destructor
    ~BremsstrahlungProcess();

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
    BremsstrahlungProcess& operator=(const BremsstrahlungProcess& right)
        = delete;
    BremsstrahlungProcess(const BremsstrahlungProcess&) = delete;

  private:
    ModelSelection model_selection_;
};

//---------------------------------------------------------------------------//
} // namespace geant_exporter
