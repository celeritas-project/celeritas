//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantPhysicsList.cc
//---------------------------------------------------------------------------//
#include "GeantPhysicsList.hh"

#include <memory>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4ComptonScattering.hh>
#include <G4CoulombScattering.hh>
#include <G4Electron.hh>
#include <G4EmParameters.hh>
#include <G4Gamma.hh>
#include <G4GammaConversion.hh>
#include <G4GammaGeneralProcess.hh>
#include <G4GoudsmitSaundersonMscModel.hh>
#include <G4LivermorePhotoElectricModel.hh>
#include <G4LossTableManager.hh>
#include <G4MollerBhabhaModel.hh>
#include <G4PairProductionRelModel.hh>
#include <G4PhotoElectricEffect.hh>
#include <G4PhysicsListHelper.hh>
#include <G4Positron.hh>
#include <G4ProcessType.hh>
#include <G4Proton.hh>
#include <G4RayleighScattering.hh>
#include <G4UrbanMscModel.hh>
#include <G4WentzelVIModel.hh>
#include <G4eCoulombScatteringModel.hh>
#include <G4eIonisation.hh>
#include <G4eMultipleScattering.hh>
#include <G4eplusAnnihilation.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"

#include "GeantBremsstrahlungProcess.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with physics options.
 */
GeantPhysicsList::GeantPhysicsList(Options const& options) : options_(options)
{
    // Set EM options using limits from G4EmParameters
    auto& em_parameters = *G4EmParameters::Instance();
    CELER_VALIDATE(options_.em_bins_per_decade >= 5,
                   << "number of EM bins per decade="
                   << options.em_bins_per_decade << " (must be at least 5)");

    em_parameters.SetNumberOfBinsPerDecade(options.em_bins_per_decade);
    em_parameters.SetLossFluctuations(options.eloss_fluctuation);
    em_parameters.SetMinEnergy(value_as<units::MevEnergy>(options.min_energy)
                               * CLHEP::MeV);
    em_parameters.SetMaxEnergy(value_as<units::MevEnergy>(options.max_energy)
                               * CLHEP::MeV);
    em_parameters.SetLPM(options.lpm);
    em_parameters.SetFluo(options.relaxation != RelaxationSelection::none);
    em_parameters.SetAuger(options.relaxation == RelaxationSelection::all);
    em_parameters.SetIntegral(options.integral_approach);
    em_parameters.SetLinearLossLimit(options.linear_loss_limit);
    em_parameters.SetMscRangeFactor(options.msc_range_factor);
    em_parameters.SetMscSafetyFactor(options.msc_safety_factor);
    em_parameters.SetMscLambdaLimit(options.msc_lambda_limit * CLHEP::cm);
    em_parameters.SetLowestElectronEnergy(
        value_as<units::MevEnergy>(options.lowest_electron_energy)
        * CLHEP::MeV);
    if (options_.msc == MscModelSelection::urban_extended)
    {
        CELER_LOG(debug) << "Extended low-energy MSC limit to 100 TeV";
        em_parameters.SetMscEnergyLimit(100 * CLHEP::TeV);
    }

    int verb = options_.verbose ? 1 : 0;
    this->SetVerboseLevel(verb);
    em_parameters.SetVerbose(verb);
}

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
 * Currently only instantiating e+, e-, gamma, and proton (the latter is needed
 * for msc)
 */
void GeantPhysicsList::ConstructParticle()
{
    G4Gamma::GammaDefinition();
    G4Electron::ElectronDefinition();
    G4Positron::PositronDefinition();
    if (options_.msc != MscModelSelection::none)
    {
        G4Proton::ProtonDefinition();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Build list of available processes and models.
 */
void GeantPhysicsList::ConstructProcess()
{
    // Applies to all constructed particles
    G4VUserPhysicsList::AddTransportation();

    // Add E.M. processes for photons, electrons, and positrons
    this->add_gamma_processes();
    this->add_e_processes(G4Electron::Electron());
    this->add_e_processes(G4Positron::Positron());
}

//---------------------------------------------------------------------------//
// PRIVATE
//---------------------------------------------------------------------------//
/*!
 * Add EM processes for photons.
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
 */
void GeantPhysicsList::add_gamma_processes()
{
    auto* physics_list = G4PhysicsListHelper::GetPhysicsListHelper();
    auto* gamma = G4Gamma::Gamma();

    // Option to create GammaGeneral for performance/robustness
    std::unique_ptr<G4GammaGeneralProcess> ggproc;
    if (options_.gamma_general)
    {
        ggproc = std::make_unique<G4GammaGeneralProcess>();
    }

    auto add_process = [&ggproc, physics_list, gamma](G4VEmProcess* p) {
        if (ggproc)
        {
            ggproc->AddEmProcess(p);
        }
        else
        {
            physics_list->RegisterProcess(p, gamma);
        }
    };

    if (true)
    {
        // Compton Scattering: G4KleinNishinaCompton
        auto compton_scattering = std::make_unique<G4ComptonScattering>();
        add_process(compton_scattering.release());
        CELER_LOG(debug) << "Loaded Compton scattering with "
                            "G4KleinNishinaCompton";
    }

    if (true)
    {
        // Photoelectric effect: G4LivermorePhotoElectricModel
        auto pe = std::make_unique<G4PhotoElectricEffect>();
        pe->SetEmModel(new G4LivermorePhotoElectricModel());
        add_process(pe.release());
        CELER_LOG(debug) << "Loaded photoelectric effect with "
                            "G4LivermorePhotoElectricModel";
    }

    if (options_.rayleigh_scattering)
    {
        // Rayleigh: G4LivermoreRayleighModel
        add_process(new G4RayleighScattering());
        CELER_LOG(debug) << "Loaded Rayleigh scattering with "
                            "G4LivermoreRayleighModel";
    }

    if (true)
    {
        // Gamma conversion: G4PairProductionRelModel
        auto gamma_conversion = std::make_unique<G4GammaConversion>();
        gamma_conversion->SetEmModel(new G4PairProductionRelModel());
        add_process(gamma_conversion.release());
        CELER_LOG(debug) << "Loaded gamma conversion with "
                            "G4PairProductionRelModel";
    }

    if (ggproc)
    {
        CELER_LOG(debug) << "Registered G4GammaGeneralProcess";
        G4LossTableManager::Instance()->SetGammaGeneralProcess(ggproc.get());
        physics_list->RegisterProcess(ggproc.release(), gamma);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Add EM processes for electrons and positrons.
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
 *   \c GeantBremsstrahlungProcess::ModelSelection and need to be updated
 *   accordingly.
 * - Coulomb scattering and multiple scattering (high E) are currently
 *   disabled.
 */
void GeantPhysicsList::add_e_processes(G4ParticleDefinition* p)
{
    auto* physics_list = G4PhysicsListHelper::GetPhysicsListHelper();

    if (p == G4Positron::Positron())
    {
        // e+e- annihilation: G4eeToTwoGammaModel
        physics_list->RegisterProcess(new G4eplusAnnihilation(), p);

        CELER_LOG(debug) << "Loaded pair annihilation with "
                            "G4eplusAnnihilation";
    }

    if (true)
    {
        // e-e+ ionization: G4MollerBhabhaModel
        auto ionization = std::make_unique<G4eIonisation>();
        ionization->SetEmModel(new G4MollerBhabhaModel());
        physics_list->RegisterProcess(ionization.release(), p);

        CELER_LOG(debug) << "Loaded ionization with G4MollerBhabhaModel";
    }

    if (true)
    {
        physics_list->RegisterProcess(
            new GeantBremsstrahlungProcess(options_.brems), p);

        auto msg = CELER_LOG(debug);
        msg << "Loaded Bremsstrahlung with ";
        switch (options_.brems)
        {
            case BremsModelSelection::seltzer_berger:
                msg << "G4SeltzerBergerModel";
                break;
            case BremsModelSelection::relativistic:
                msg << "G4eBremsstrahlungRelModel";
                break;
            case BremsModelSelection::all:
                msg << "G4SeltzerBergerModel and G4eBremsstrahlungRelModel";
                break;
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }

    if (options_.coulomb_scattering)
    {
        // Coulomb scattering: G4eCoulombScatteringModel
        double msc_energy_limit = G4EmParameters::Instance()->MscEnergyLimit();

        auto process = std::make_unique<G4CoulombScattering>();
        auto model = std::make_unique<G4eCoulombScatteringModel>();
        process->SetMinKinEnergy(msc_energy_limit);
        model->SetLowEnergyLimit(msc_energy_limit);
        model->SetActivationLowEnergyLimit(msc_energy_limit);
        process->SetEmModel(model.release());
        physics_list->RegisterProcess(process.release(), p);
    }

    if (options_.msc != MscModelSelection::none)
    {
        // Multiple scattering: Urban (low E) and WentzelVI (high E) models
        double msc_energy_limit = G4EmParameters::Instance()->MscEnergyLimit();

        auto process = std::make_unique<G4eMultipleScattering>();

        if (options_.msc == MscModelSelection::urban
            || options_.msc == MscModelSelection::urban_extended
            || options_.msc == MscModelSelection::urban_wentzel)
        {
            auto model = std::make_unique<G4UrbanMscModel>();
            model->SetHighEnergyLimit(msc_energy_limit);
            process->SetEmModel(model.release());

            CELER_LOG(debug) << "Loaded low-energy multiple scattering with "
                                "G4UrbanMscModel";
        }

        if (options_.msc == MscModelSelection::wentzel_vi
            || options_.msc == MscModelSelection::urban_wentzel)
        {
            auto model = std::make_unique<G4WentzelVIModel>();
            model->SetLowEnergyLimit(msc_energy_limit);
            process->SetEmModel(model.release());

            CELER_LOG(debug) << "Loaded high-energy multiple scattering with "
                                "G4WentzelVIModel";
        }

        if (options_.msc == MscModelSelection::goudsmit_saunderson)
        {
            auto model = std::make_unique<G4GoudsmitSaundersonMscModel>();
            process->SetEmModel(model.release());

            CELER_LOG(debug) << "Loaded multiple scattering with "
                                "G4GoudsmitSaundersonMscModel";
        }

        physics_list->RegisterProcess(process.release(), p);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
