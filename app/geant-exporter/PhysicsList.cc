//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsList.cc
//---------------------------------------------------------------------------//
#include "PhysicsList.hh"

#include <G4ProcessManager.hh>
#include <G4SystemOfUnits.hh>
#include <G4EmBuilder.hh>
#include <G4PhysicsListHelper.hh>

#include <G4ComptonScattering.hh>
#include <G4KleinNishinaModel.hh>

#include <G4PhotoElectricEffect.hh>
#include <G4LivermorePhotoElectricModel.hh>

// Not from Geant4
#include "BremsstrahlungProcess.hh"

#include <G4GammaConversion.hh>
#include <G4BetheHeitlerModel.hh>

#include <G4CoulombScattering.hh>
#include <G4eCoulombScatteringModel.hh>

#include <G4eIonisation.hh>
#include <G4MollerBhabhaModel.hh>

#include <G4RayleighScattering.hh>
#include <G4LivermoreRayleighModel.hh>

#include <G4eplusAnnihilation.hh>
#include <G4eeToTwoGammaModel.hh>

#include <G4eMultipleScattering.hh>
#include <G4UrbanMscModel.hh>
#include <G4WentzelVIModel.hh>

namespace geant_exporter
{
//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
PhysicsList::PhysicsList() : G4VUserPhysicsList()
{
    // Manually select the physics table binning
    // G4EmParameters* em_parameters = G4EmParameters::Instance();
    // em_parameters->SetNumberOfBins(10);
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
PhysicsList::~PhysicsList() = default;

//---------------------------------------------------------------------------//
/*!
 * Build list of available particles.
 *
 * The minimal E.M. set can be built by using
 * \c G4EmBuilder::ConstructMinimalEmSet();
 * and includes gamma, e+, e-, mu+, mu-, pi+, pi-, K+, K-, p, pbar, deuteron,
 * triton, He3, alpha, and generic ion, along with Geant4's pseudo-particles
 * geantino and charged geantino.
 *
 * Currently only instantiating e+, e-, and gamma, as in Celeritas.
 */
void PhysicsList::ConstructParticle()
{
    G4Gamma::GammaDefinition();
    G4Electron::ElectronDefinition();
    G4Positron::PositronDefinition();
}

//---------------------------------------------------------------------------//
/*!
 * Build list of available processes and models.
 */
void PhysicsList::ConstructProcess()
{
    // Applies to all constructed particles
    G4VUserPhysicsList::AddTransportation();

    // Add E.M. processes for photons, electrons, and positrons
    this->add_gamma_processes();
    this->add_e_processes();
}
//---------------------------------------------------------------------------//
// PRIVATE
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Add E.M. processes for photons.
 *
 * | Processes            | Model classes                 |
 * | -------------------- | ----------------------------- |
 * | Compton scattering   | G4KleinNishinaModel           |
 * | Photoelectric effect | G4LivermorePhotoElectricModel |
 * | Rayleigh scattering  | G4LivermoreRayleighModel      |
 * | Gamma conversion     | G4BetheHeitlerModel           |
 */
void PhysicsList::add_gamma_processes()
{
    auto       physics_list = G4PhysicsListHelper::GetPhysicsListHelper();
    const auto gamma        = G4Gamma::Gamma();

    if (true)
    {
        // Compton Scattering: G4KleinNishinaModel
        auto compton_scattering = std::make_unique<G4ComptonScattering>();
        compton_scattering->SetEmModel(new G4KleinNishinaModel());
        physics_list->RegisterProcess(compton_scattering.release(), gamma);
    }

    if (true)
    {
        // Photoelectric effect: G4LivermorePhotoElectricModel
        auto photoelectrict_effect = std::make_unique<G4PhotoElectricEffect>();
        photoelectrict_effect->SetEmModel(new G4LivermorePhotoElectricModel());
        physics_list->RegisterProcess(photoelectrict_effect.release(), gamma);
    }

    if (true)
    {
        // Rayleigh: G4LivermoreRayleighModel
        physics_list->RegisterProcess(new G4RayleighScattering(), gamma);
    }

    if (true)
    {
        // Gamma conversion: G4BetheHeitlerModel
        auto gamma_conversion = std::make_unique<G4GammaConversion>();
        gamma_conversion->SetEmModel(new G4BetheHeitlerModel());
        physics_list->RegisterProcess(gamma_conversion.release(), gamma);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Add E.M. processes for electrons and positrons.
 *
 * | Processes                    | Model classes             |
 * | ---------------------------- | --------------------------|
 * | Pair annihilation            | G4eeToTwoGammaModel       |
 * | Ionization                   | G4MollerBhabhaModel       |
 * | Bremsstrahlung (low E)       | G4SeltzerBergerModel      |
 * | Bremsstrahlung (high E)      | G4eBremsstrahlungRelModel |
 * | Coulomb scattering           | G4eCoulombScatteringModel |
 * | Multiple scattering (low E)  | G4UrbanMscModel           |
 * | Multiple scattering (high E) | G4WentzelVIModel          |
 *
 * \note
 * - Bremsstrahlung models are selected manually at compile time using
 *   \c BremsModelSelection and need to be updated accordingly.
 * - Coulomb and multiple scatterings are currently disabled.
 */
void PhysicsList::add_e_processes()
{
    auto       physics_list = G4PhysicsListHelper::GetPhysicsListHelper();
    const auto electron     = G4Electron::Electron();
    const auto positron     = G4Positron::Positron();

    if (true)
    {
        // e+e- annihilation: G4eeToTwoGammaModel
        physics_list->RegisterProcess(new G4eplusAnnihilation(), positron);
    }

    if (true)
    {
        // e-e+ ionization: G4MollerBhabhaModel
        auto electron_ionization = std::make_unique<G4eIonisation>();
        auto positron_ionization = std::make_unique<G4eIonisation>();
        electron_ionization->SetEmModel(new G4MollerBhabhaModel());
        positron_ionization->SetEmModel(new G4MollerBhabhaModel());
        physics_list->RegisterProcess(electron_ionization.release(), electron);
        physics_list->RegisterProcess(positron_ionization.release(), positron);
    }

    if (true)
    {
        // Bremsstrahlung: G4SeltzerBergerModel + G4eBremsstrahlungRelModel
        // Currently only using Seltzer-Berger
        auto models = BremsModelSelection::seltzer_berger;

        auto electron_brems = std::make_unique<BremsstrahlungProcess>(models);
        auto positron_brems = std::make_unique<BremsstrahlungProcess>(models);
        physics_list->RegisterProcess(electron_brems.release(), electron);
        physics_list->RegisterProcess(positron_brems.release(), positron);
    }

    // DISABLED
    if (false)
    {
        // Coulomb scattering: G4eCoulombScatteringModel
        double msc_energy_limit = G4EmParameters::Instance()->MscEnergyLimit();

        // Electron
        auto coulomb_scat_electron = std::make_unique<G4CoulombScattering>();
        auto coulomb_model_electron
            = std::make_unique<G4eCoulombScatteringModel>();
        coulomb_scat_electron->SetMinKinEnergy(msc_energy_limit);
        coulomb_model_electron->SetLowEnergyLimit(msc_energy_limit);
        coulomb_model_electron->SetActivationLowEnergyLimit(msc_energy_limit);
        coulomb_scat_electron->SetEmModel(coulomb_model_electron.release());
        physics_list->RegisterProcess(coulomb_scat_electron.release(),
                                      electron);

        // Positron
        auto coulomb_scat_positron = std::make_unique<G4CoulombScattering>();
        auto coulomb_model_positron
            = std::make_unique<G4eCoulombScatteringModel>();
        coulomb_scat_positron->SetMinKinEnergy(msc_energy_limit);
        coulomb_model_positron->SetLowEnergyLimit(msc_energy_limit);
        coulomb_model_positron->SetActivationLowEnergyLimit(msc_energy_limit);
        coulomb_scat_positron->SetEmModel(coulomb_model_positron.release());
        physics_list->RegisterProcess(coulomb_scat_positron.release(),
                                      positron);
    }

    // DISABLED
    if (false)
    {
        // Multiple scattering: Urban (low E) and WentzelVI (high E) models
        double msc_energy_limit = G4EmParameters::Instance()->MscEnergyLimit();

        // Electron
        auto msc_electron       = std::make_unique<G4eMultipleScattering>();
        auto urban_msc_electron = std::make_unique<G4UrbanMscModel>();
        auto wentzelvi_msc_electron = std::make_unique<G4WentzelVIModel>();
        urban_msc_electron->SetHighEnergyLimit(msc_energy_limit);
        wentzelvi_msc_electron->SetLowEnergyLimit(msc_energy_limit);
        msc_electron->SetEmModel(urban_msc_electron.release());
        msc_electron->SetEmModel(wentzelvi_msc_electron.release());
        physics_list->RegisterProcess(msc_electron.release(), electron);

        // Positron
        auto msc_positron       = std::make_unique<G4eMultipleScattering>();
        auto urban_msc_positron = std::make_unique<G4UrbanMscModel>();
        auto wentzelvi_msc_positron = std::make_unique<G4WentzelVIModel>();
        urban_msc_positron->SetHighEnergyLimit(msc_energy_limit);
        wentzelvi_msc_positron->SetLowEnergyLimit(msc_energy_limit);
        msc_positron->SetEmModel(urban_msc_positron.release());
        msc_positron->SetEmModel(wentzelvi_msc_positron.release());
        physics_list->RegisterProcess(msc_positron.release(), positron);
    }
}

//---------------------------------------------------------------------------//
} // namespace geant_exporter
