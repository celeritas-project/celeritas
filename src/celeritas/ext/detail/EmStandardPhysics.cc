//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/EmStandardPhysics.cc
//---------------------------------------------------------------------------//
#include "EmStandardPhysics.hh"

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
#include <G4MuBremsstrahlung.hh>
#include <G4MuIonisation.hh>
#include <G4MuMultipleScattering.hh>
#include <G4MuPairProduction.hh>
#include <G4MuonMinus.hh>
#include <G4MuonPlus.hh>
#include <G4PairProductionRelModel.hh>
#include <G4PhotoElectricEffect.hh>
#include <G4PhysicsListHelper.hh>
#include <G4Positron.hh>
#include <G4ProcessManager.hh>
#include <G4ProcessType.hh>
#include <G4Proton.hh>
#include <G4RayleighScattering.hh>
#include <G4UrbanMscModel.hh>
#include <G4Version.hh>
#include <G4WentzelVIModel.hh>
#include <G4eCoulombScatteringModel.hh>
#include <G4eIonisation.hh>
#include <G4eMultipleScattering.hh>
#include <G4eplusAnnihilation.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Quantities.hh"

#include "GeantBremsstrahlungProcess.hh"
#include "../GeantPhysicsOptions.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Safely switch from MscStepLimitAlgorithm to G4MscStepLimitType.
 */
G4MscStepLimitType
from_msc_step_algorithm(MscStepLimitAlgorithm const& msc_step_algorithm)
{
    switch (msc_step_algorithm)
    {
        case MscStepLimitAlgorithm::minimal:
            return G4MscStepLimitType::fMinimal;
        case MscStepLimitAlgorithm::safety:
            return G4MscStepLimitType::fUseSafety;
        case MscStepLimitAlgorithm::safety_plus:
            return G4MscStepLimitType::fUseSafetyPlus;
        case MscStepLimitAlgorithm::distance_to_boundary:
            return G4MscStepLimitType::fUseDistanceToBoundary;
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Safely switch from NuclearFormFactorType to G4NuclearFormfactorType.
 */
G4NuclearFormfactorType
from_form_factor_type(NuclearFormFactorType const& form_factor)
{
    switch (form_factor)
    {
        case NuclearFormFactorType::none:
            return G4NuclearFormfactorType::fNoneNF;
        case NuclearFormFactorType::exponential:
            return G4NuclearFormfactorType::fExponentialNF;
        case NuclearFormFactorType::gaussian:
            return G4NuclearFormfactorType::fGaussianNF;
        case NuclearFormFactorType::flat:
            return G4NuclearFormfactorType::fFlatNF;
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct with physics options.
 */
EmStandardPhysics::EmStandardPhysics(Options const& options)
    : options_(options)
{
    // Set EM options using limits from G4EmParameters
    auto& em_parameters = *G4EmParameters::Instance();
    CELER_VALIDATE(options_.em_bins_per_decade >= 5,
                   << "number of EM bins per decade="
                   << options.em_bins_per_decade << " (must be at least 5)");

    em_parameters.SetNumberOfBinsPerDecade(options.em_bins_per_decade);
    em_parameters.SetLossFluctuations(options.eloss_fluctuation);
    em_parameters.SetMinEnergy(value_as<Options::MevEnergy>(options.min_energy)
                               * CLHEP::MeV);
    em_parameters.SetMaxEnergy(value_as<Options::MevEnergy>(options.max_energy)
                               * CLHEP::MeV);
    em_parameters.SetLPM(options.lpm);
    em_parameters.SetFluo(options.relaxation != RelaxationSelection::none);
    em_parameters.SetAuger(options.relaxation == RelaxationSelection::all);
    em_parameters.SetIntegral(options.integral_approach);
    em_parameters.SetLinearLossLimit(options.linear_loss_limit);
    em_parameters.SetNuclearFormfactorType(
        from_form_factor_type(options.form_factor));
    em_parameters.SetMscStepLimitType(
        from_msc_step_algorithm(options.msc_step_algorithm));
    em_parameters.SetMscMuHadStepLimitType(
        from_msc_step_algorithm(options.msc_muhad_step_algorithm));
    em_parameters.SetLateralDisplacement(options.msc_displaced);
    em_parameters.SetMuHadLateralDisplacement(options.msc_muhad_displaced);
    em_parameters.SetMscRangeFactor(options.msc_range_factor);
    em_parameters.SetMscMuHadRangeFactor(options.msc_muhad_range_factor);
#if G4VERSION_NUMBER >= 1060
    using ClhepLen = Quantity<units::ClhepTraits::Length, double>;

    // Customizable MSC safety factor/lambda limit were added in
    // emutils-V10-05-18
    em_parameters.SetMscSafetyFactor(options.msc_safety_factor);
    em_parameters.SetMscLambdaLimit(
        native_value_to<ClhepLen>(options.msc_lambda_limit).value());
#endif
    em_parameters.SetMscThetaLimit(options.msc_theta_limit);
    em_parameters.SetLowestElectronEnergy(
        value_as<Options::MevEnergy>(options.lowest_electron_energy)
        * CLHEP::MeV);
    em_parameters.SetLowestMuHadEnergy(
        value_as<Options::MevEnergy>(options.lowest_muhad_energy) * CLHEP::MeV);
    em_parameters.SetApplyCuts(options.apply_cuts);
    em_parameters.SetVerbose(options.verbose);
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
 * Currently only instantiating e+, e-, gamma, mu-, mu+, and proton (the latter
 * is needed for msc)
 */
void EmStandardPhysics::ConstructParticle()
{
    G4Gamma::GammaDefinition();
    G4Electron::ElectronDefinition();
    G4Positron::PositronDefinition();
    if (options_.muon)
    {
        G4MuonMinus::MuonMinus();
        G4MuonPlus::MuonPlus();
    }
    if (options_.msc != MscModelSelection::none || options_.coulomb_scattering)
    {
        G4Proton::ProtonDefinition();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Build list of available processes and models.
 */
void EmStandardPhysics::ConstructProcess()
{
    // Add E.M. processes for photons, electrons, and positrons
    this->add_gamma_processes();
    this->add_e_processes(G4Electron::Electron());
    this->add_e_processes(G4Positron::Positron());
    if (options_.muon)
    {
        this->add_mu_processes(G4MuonMinus::MuonMinus());
        this->add_mu_processes(G4MuonPlus::MuonPlus());
    }
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
void EmStandardPhysics::add_gamma_processes()
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

    if (options_.compton_scattering)
    {
        // Compton Scattering: G4KleinNishinaCompton
        auto compton_scattering = std::make_unique<G4ComptonScattering>();
        add_process(compton_scattering.release());
        CELER_LOG(debug) << "Loaded Compton scattering with "
                            "G4KleinNishinaCompton";
    }

    if (options_.photoelectric)
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
        auto rayl = std::make_unique<G4RayleighScattering>();
        if (G4VERSION_NUMBER >= 1110)
        {
            // Revert MinKinEnergyPrim (150 keV) to its Geant 11.0 value so
            // that the lower and energy grids have the same log spacing
            rayl->SetMinKinEnergyPrim(100 * CLHEP::keV);
        }
        add_process(rayl.release());
        CELER_LOG(debug) << "Loaded Rayleigh scattering with "
                            "G4LivermoreRayleighModel";
    }

    if (options_.gamma_conversion)
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
 * | Processes                    | Model classes                |
 * | ---------------------------- | ---------------------------- |
 * | Pair annihilation            | G4eeToTwoGammaModel          |
 * | Ionization                   | G4MollerBhabhaModel          |
 * | Bremsstrahlung (low E)       | G4SeltzerBergerModel         |
 * | Bremsstrahlung (high E)      | G4eBremsstrahlungRelModel    |
 * | Coulomb scattering           | G4eCoulombScatteringModel    |
 * | Multiple scattering (low E)  | G4UrbanMscModel              |
 * | Multiple scattering (low E)  | G4GoudsmitSaundersonMscModel |
 * | Multiple scattering (high E) | G4WentzelVIModel             |
 *
 * \note
 * - Coulomb scattering and multiple scattering (high E) are currently
 *   disabled.
 */
void EmStandardPhysics::add_e_processes(G4ParticleDefinition* p)
{
    auto* physics_list = G4PhysicsListHelper::GetPhysicsListHelper();

    if (options_.annihilation && p == G4Positron::Positron())
    {
        // e+e- annihilation: G4eeToTwoGammaModel
        physics_list->RegisterProcess(new G4eplusAnnihilation(), p);

        CELER_LOG(debug) << "Loaded pair annihilation with "
                            "G4eplusAnnihilation";
    }

    if (options_.ionization)
    {
        // e-e+ ionization: G4MollerBhabhaModel
        auto ionization = std::make_unique<G4eIonisation>();
        ionization->SetEmModel(new G4MollerBhabhaModel());
        physics_list->RegisterProcess(ionization.release(), p);

        CELER_LOG(debug) << "Loaded ionization with G4MollerBhabhaModel";
    }

    if (options_.brems != BremsModelSelection::none)
    {
        physics_list->RegisterProcess(
            new GeantBremsstrahlungProcess(options_.brems), p);

        if (!options_.ionization)
        {
            // If ionization is turned off, activate the along-step "do it" for
            // bremsstrahlung *after* the process has been registered and set
            // the order to be the same as the default post-step order. See \c
            // G4PhysicsListHelper and the ordering parameter table for more
            // information on which "do its" are activated for each process and
            // the default process ordering.
            auto* process_manager = p->GetProcessManager();
            CELER_ASSERT(process_manager);
            auto* bremsstrahlung = dynamic_cast<GeantBremsstrahlungProcess*>(
                process_manager->GetProcess("eBrem"));
            CELER_ASSERT(bremsstrahlung);
            auto order = process_manager->GetProcessOrdering(
                bremsstrahlung, G4ProcessVectorDoItIndex::idxPostStep);
            process_manager->SetProcessOrdering(
                bremsstrahlung, G4ProcessVectorDoItIndex::idxAlongStep, order);

            // Let this process be a candidate for range limiting the step
            bremsstrahlung->SetIonisation(true);
        }

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

    using MMS = MscModelSelection;

    // Energy limit between MSC models when multiple models are used
    double msc_energy_limit = G4EmParameters::Instance()->MscEnergyLimit();
    bool set_energy_limit = options_.msc == MMS::urban_wentzelvi;

    if (options_.coulomb_scattering)
    {
        // Coulomb scattering: G4eCoulombScatteringModel
        if (options_.msc == MMS::urban)
        {
            CELER_LOG(warning)
                << "Urban multiple scattering is used for all "
                   "energies: disabling G4eCoulombScatteringModel";
        }
        else
        {
            auto process = std::make_unique<G4CoulombScattering>();
            auto model = std::make_unique<G4eCoulombScatteringModel>();
            if (set_energy_limit)
            {
                process->SetMinKinEnergy(msc_energy_limit);
                model->SetLowEnergyLimit(msc_energy_limit);
                model->SetActivationLowEnergyLimit(msc_energy_limit);
            }
            if (options_.msc == MMS::none)
            {
                G4EmParameters::Instance()->SetMscThetaLimit(0);
            }

            CELER_LOG(debug) << "Loaded single Coulomb scattering with "
                                "G4eCoulombScatteringModel from "
                             << model->LowEnergyLimit() << " MeV to "
                             << model->HighEnergyLimit() << " MeV";

            process->SetEmModel(model.release());
            physics_list->RegisterProcess(process.release(), p);
        }
    }

    if (options_.msc != MMS::none)
    {
        auto process = std::make_unique<G4eMultipleScattering>();

        if (options_.msc == MMS::urban || options_.msc == MMS::urban_wentzelvi)
        {
            // Multiple scattering: Urban
            auto model = std::make_unique<G4UrbanMscModel>();
            if (set_energy_limit)
            {
                model->SetHighEnergyLimit(msc_energy_limit);
            }

            CELER_LOG(debug) << "Loaded multiple scattering with "
                                "G4UrbanMscModel from "
                             << model->LowEnergyLimit() << " MeV to "
                             << model->HighEnergyLimit() << " MeV";

            process->SetEmModel(model.release());
        }

        if (options_.msc == MMS::wentzelvi
            || options_.msc == MMS::urban_wentzelvi)
        {
            // Multiple scattering: WentzelVI
            auto model = std::make_unique<G4WentzelVIModel>();
            if (set_energy_limit)
            {
                model->SetLowEnergyLimit(msc_energy_limit);
            }
            CELER_LOG(debug) << "Loaded multiple scattering with "
                                "G4WentzelVIModel from "
                             << model->LowEnergyLimit() << " MeV to "
                             << model->HighEnergyLimit() << " MeV";

            process->SetEmModel(model.release());
        }

        physics_list->RegisterProcess(process.release(), p);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Add EM processes for muons.
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
 * \note Currently all muon processes are disabled by default
 *
 * \note Prior to version 11.1.0, Geant4 used the \c G4BetheBlochModel for muon
 * ionization between 200 keV and 1 GeV and the \c G4MuBetheBlochModel above 1
 * GeV. Since version 11.1.0, the \c G4MuBetheBlochModel is used for all
 * energies above 200 keV.
 *
 * \todo Implement energy loss fluctuation models for muon ionization.
 */
void EmStandardPhysics::add_mu_processes(G4ParticleDefinition* p)
{
    auto* physics_list = G4PhysicsListHelper::GetPhysicsListHelper();

    if (options_.muon.pair_production)
    {
        physics_list->RegisterProcess(new G4MuPairProduction(), p);
        CELER_LOG(debug) << "Loaded muon pair production with "
                            "G4MuPairProductionModel";
    }

    if (options_.muon.ionization)
    {
        physics_list->RegisterProcess(new G4MuIonisation(), p);
        CELER_LOG(debug) << "Loaded muon ionization with G4ICRU73QOModel, "
                            "G4BraggModel, and G4MuBetheBlochModel";
    }

    if (options_.muon.bremsstrahlung)
    {
        physics_list->RegisterProcess(new G4MuBremsstrahlung(), p);
        CELER_LOG(debug) << "Loaded muon bremsstrahlung with "
                            "G4MuBremsstrahlungModel";
    }

    if (options_.muon.coulomb)
    {
        physics_list->RegisterProcess(new G4CoulombScattering(), p);
        CELER_LOG(debug) << "Loaded muon Coulomb scattering with "
                            "G4eCoulombScatteringModel";
    }

    if (options_.muon.msc != MscModelSelection::none)
    {
        auto process = std::make_unique<G4MuMultipleScattering>();
        if (options_.muon.msc == MscModelSelection::wentzelvi)
        {
            process->SetEmModel(new G4WentzelVIModel());
            CELER_LOG(debug) << "Loaded muon multiple scattering with "
                                "G4WentzelVIModel";
        }
        else if (options_.muon.msc == MscModelSelection::urban)
        {
            process->SetEmModel(new G4UrbanMscModel());
            CELER_LOG(debug) << "Loaded muon multiple scattering with "
                                "G4UrbanMscModel";
        }
        else
        {
            CELER_VALIDATE(false,
                           << "unsupported muon multiple scattering model "
                              "selection '"
                           << to_cstring(options_.muon.msc) << "'");
        }
        physics_list->RegisterProcess(process.release(), p);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
