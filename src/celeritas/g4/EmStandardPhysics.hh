//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/EmStandardPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VPhysicsConstructor.hh>

#include "celeritas/ext/GeantPhysicsOptions.hh"

class G4ParticleDefinition;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas-supported Geant4 EM physics processes.
 *
 * This is a configurable plug-in replacement for \c G4EmStandardPhysics.
 * Limitations:
 * - No support for generic ions
 * - No hadronic EM interactions
 * - Limited support for muons (currently disabled by default)
 * - Coulomb scattering and multiple scattering (high E) are currently
 *   disabled.
 *
 * Electron/positron processes:
 *
 * | Processes                    | Model classes                |
 * | ---------------------------- | ---------------------------- |
 * | Pair annihilation            | G4eeToTwoGammaModel          |
 * | Ionization                   | G4MollerBhabhaModel          |
 * | Bremsstrahlung (low E)       | G4SeltzerBergerModel         |
 * | Bremsstrahlung (high E)      | G4eBremsstrahlungRelModel    |
 * | Coulomb scattering           | G4eCoulombScatteringModel    |
 * | Multiple scattering (low E)  | G4UrbanMscModel              |
 * | Multiple scattering (high E) | G4WentzelVIModel             |
 *
 * Gamma processes:
 *
 * | Processes            | Model classes                 |
 * | -------------------- | ----------------------------- |
 * | Compton scattering   | G4KleinNishinaCompton         |
 * | Photoelectric effect | G4LivermorePhotoElectricModel |
 * | Rayleigh scattering  | G4LivermoreRayleighModel      |
 * | Gamma conversion     | G4PairProductionRelModel      |
 *
 * If the \c gamma_general option is enabled, we create a single unified
 * \c G4GammaGeneralProcess process, which embeds these other processes and
 * calculates a combined total cross section. It's faster in Geant4 but
 * shouldn't result in different answers.
 *
 * Muon processes:
 *
 * | Processes                    | Model classes                |
 * | ---------------------------- | ---------------------------- |
 * | Pair production              | G4MuPairProductionModel      |
 * | Ionization (low E, mu-)      | G4ICRU73QOModel              |
 * | Ionization (low E, mu+)      | G4BraggModel                 |
 * | Ionization (high E)          | G4MuBetheBlochModel          |
 * | Bremsstrahlung               | G4MuBremsstrahlungModel      |
 * | Coulomb scattering           | G4eCoulombScatteringModel    |
 * | Multiple scattering          | G4WentzelVIModel             |
 *
 * \note Prior to version 11.1.0, Geant4 used the \c G4BetheBlochModel for muon
 * ionization between 200 keV and 1 GeV and the \c G4MuBetheBlochModel above 1
 * GeV. Since version 11.1.0, the \c G4MuBetheBlochModel is used for all
 * energies above 200 keV.
 */
class EmStandardPhysics : public G4VPhysicsConstructor
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantPhysicsOptions;
    //!@}

  public:
    // Set up during construction
    explicit EmStandardPhysics(Options const& options);

    // Set up minimal EM particle list
    void ConstructParticle() override;
    // Set up process list
    void ConstructProcess() override;

  private:
    Options options_;

    // Add EM processes for photons
    void add_gamma_processes();
    // Add EM processes for electrons and positrons
    void add_e_processes(G4ParticleDefinition* p);
    // Add EM processes for muons
    void add_mu_processes(G4ParticleDefinition* p);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
