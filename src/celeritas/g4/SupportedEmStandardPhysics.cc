//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/SupportedEmStandardPhysics.cc
//---------------------------------------------------------------------------//
#include "SupportedEmStandardPhysics.hh"

#include <memory>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4BuilderType.hh>
#include <G4ComptonScattering.hh>
#include <G4CoulombScattering.hh>
#include <G4Electron.hh>
#include <G4EmParameters.hh>
#include <G4Gamma.hh>
#include <G4GammaConversion.hh>
#include <G4GammaGeneralProcess.hh>
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
#include <G4ParticleDefinition.hh>
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
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"

#include "detail/GeantBremsstrahlungProcess.hh"

namespace celeritas
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
SupportedEmStandardPhysics::SupportedEmStandardPhysics(Options const& options)
    : G4VPhysicsConstructor("CelerEmStandard", bElectromagnetic)
    , options_(options)
{
    CELER_LOG(debug) << "Setting EM parameters";

    // Set EM options using limits from G4EmParameters
    auto& em_params = *G4EmParameters::Instance();
    CELER_VALIDATE(options_.em_bins_per_decade >= 5,
                   << "number of EM bins per decade="
                   << options.em_bins_per_decade << " (must be at least 5)");

    em_params.SetNumberOfBinsPerDecade(options.em_bins_per_decade);
    em_params.SetLossFluctuations(options.eloss_fluctuation);
    em_params.SetMinEnergy(value_as<Options::MevEnergy>(options.min_energy)
                           * CLHEP::MeV);
    em_params.SetMaxEnergy(value_as<Options::MevEnergy>(options.max_energy)
                           * CLHEP::MeV);
    em_params.SetLPM(options.lpm);
    em_params.SetFluo(options.relaxation != RelaxationSelection::none);
    em_params.SetAuger(options.relaxation == RelaxationSelection::all);
    em_params.SetIntegral(options.integral_approach);
    em_params.SetLinearLossLimit(options.linear_loss_limit);
    em_params.SetNuclearFormfactorType(
        from_form_factor_type(options.form_factor));
    em_params.SetMscStepLimitType(
        from_msc_step_algorithm(options.msc_step_algorithm));
    em_params.SetMscMuHadStepLimitType(
        from_msc_step_algorithm(options.msc_muhad_step_algorithm));
    em_params.SetLateralDisplacement(options.msc_displaced);
    em_params.SetMuHadLateralDisplacement(options.msc_muhad_displaced);
    em_params.SetMscRangeFactor(options.msc_range_factor);
    em_params.SetMscMuHadRangeFactor(options.msc_muhad_range_factor);
#if G4VERSION_NUMBER >= 1060
    using ClhepLen = Quantity<units::ClhepTraits::Length, double>;

    // Customizable MSC safety factor/lambda limit were added in
    // emutils-V10-05-18
    em_params.SetMscSafetyFactor(options.msc_safety_factor);
    em_params.SetMscLambdaLimit(
        native_value_to<ClhepLen>(options.msc_lambda_limit).value());
#endif
    em_params.SetMscThetaLimit(options.msc_theta_limit);
    em_params.SetLowestElectronEnergy(
        value_as<Options::MevEnergy>(options.lowest_electron_energy)
        * CLHEP::MeV);
    em_params.SetLowestMuHadEnergy(
        value_as<Options::MevEnergy>(options.lowest_muhad_energy) * CLHEP::MeV);
    em_params.SetApplyCuts(options.apply_cuts);
    em_params.SetVerbose(options.verbose);
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
 * This method is called when the physics list is provided to the run manager.
 */
void SupportedEmStandardPhysics::ConstructParticle()
{
    CELER_LOG(debug) << "Constructing particles";

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
void SupportedEmStandardPhysics::ConstructProcess()
{
    CELER_LOG_LOCAL(debug) << "Constructing processes";

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
 */
void SupportedEmStandardPhysics::add_gamma_processes()
{
    auto& ph = *G4PhysicsListHelper::GetPhysicsListHelper();
    auto& em_params = *G4EmParameters::Instance();

#if G4VERSION_NUMBER >= 1060
    if (em_params.EnablePolarisation())
    {
        CELER_NOT_IMPLEMENTED("polarized gamma processes");
    }
#else
    CELER_DISCARD(em_params);
#endif

    auto* gamma = G4Gamma::Gamma();
    // Option to create GammaGeneral for performance/robustness
    std::unique_ptr<G4GammaGeneralProcess> ggproc;
    if (options_.gamma_general)
    {
        ggproc = std::make_unique<G4GammaGeneralProcess>();
    }

    auto add_process = [&ggproc, &ph, gamma](G4VEmProcess* p) {
        if (ggproc)
        {
            ggproc->AddEmProcess(p);
        }
        else
        {
            ph.RegisterProcess(p, gamma);
        }
    };

    if (options_.compton_scattering)
    {
        // Compton Scattering: G4KleinNishinaCompton
        auto compton_scattering = std::make_unique<G4ComptonScattering>();
        add_process(compton_scattering.release());
        CELER_LOG(debug) << "Using Compton scattering with "
                            "G4KleinNishinaCompton";
    }

    if (options_.photoelectric)
    {
        // Photoelectric effect: G4LivermorePhotoElectricModel
        auto pe = std::make_unique<G4PhotoElectricEffect>();
        pe->SetEmModel(new G4LivermorePhotoElectricModel());
        add_process(pe.release());
        CELER_LOG(debug) << "Using photoelectric effect with "
                            "G4LivermorePhotoElectricModel";
    }

    if (options_.rayleigh_scattering)
    {
        // Rayleigh: G4LivermoreRayleighModel
        auto rayl = std::make_unique<G4RayleighScattering>();
        add_process(rayl.release());
        CELER_LOG(debug) << "Using Rayleigh scattering with "
                            "G4LivermoreRayleighModel";
    }

    if (options_.gamma_conversion)
    {
        // Gamma conversion: G4PairProductionRelModel
        auto gamma_conversion = std::make_unique<G4GammaConversion>();
        gamma_conversion->SetEmModel(new G4PairProductionRelModel());
        add_process(gamma_conversion.release());
        CELER_LOG(debug) << "Using gamma conversion with "
                            "G4PairProductionRelModel";
    }

    if (ggproc)
    {
        CELER_LOG(debug) << "Registered G4GammaGeneralProcess";
        G4LossTableManager::Instance()->SetGammaGeneralProcess(ggproc.get());
        ph.RegisterProcess(ggproc.release(), gamma);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Add EM processes for electrons and positrons.
 */
void SupportedEmStandardPhysics::add_e_processes(G4ParticleDefinition* p)
{
    auto& ph = *G4PhysicsListHelper::GetPhysicsListHelper();
    auto& em_params = *G4EmParameters::Instance();

    if (options_.annihilation && p == G4Positron::Positron())
    {
#if G4VERSION_NUMBER >= 1130
        if (em_params.Use3GammaAnnihilationOnFly())
        {
            CELER_NOT_IMPLEMENTED("3-gamma annihilation model");
        }
#else
        CELER_DISCARD(em_params);
#endif
        // e+e- annihilation: G4eeToTwoGammaModel
        ph.RegisterProcess(new G4eplusAnnihilation(), p);

        CELER_LOG(debug) << "Using pair annihilation with "
                            "G4eplusAnnihilation";
    }

    if (options_.ionization)
    {
        // e-e+ ionization: G4MollerBhabhaModel
        auto ionization = std::make_unique<G4eIonisation>();
        ionization->SetEmModel(new G4MollerBhabhaModel());
        ph.RegisterProcess(ionization.release(), p);

        CELER_LOG(debug) << "Using ionization with G4MollerBhabhaModel";
    }

    if (options_.brems != BremsModelSelection::none)
    {
        ph.RegisterProcess(
            new detail::GeantBremsstrahlungProcess(
                options_.brems,
                value_as<Options::MevEnergy>(options_.seltzer_berger_limit)),
            p);

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
            auto* bremsstrahlung
                = dynamic_cast<detail::GeantBremsstrahlungProcess*>(
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
        msg << "Using Bremsstrahlung with ";
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

            CELER_LOG(debug) << "Using single Coulomb scattering with "
                                "G4eCoulombScatteringModel from "
                             << model->LowEnergyLimit() << " MeV to "
                             << model->HighEnergyLimit() << " MeV";

            process->SetEmModel(model.release());
            ph.RegisterProcess(process.release(), p);
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

            CELER_LOG(debug) << "Using multiple scattering with "
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
            CELER_LOG(debug) << "Using multiple scattering with "
                                "G4WentzelVIModel from "
                             << model->LowEnergyLimit() << " MeV to "
                             << model->HighEnergyLimit() << " MeV";

            process->SetEmModel(model.release());
        }

        ph.RegisterProcess(process.release(), p);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Add EM processes for muons.
 *
 * \note Any new processes added here (i.e., when they're implemented in
 * Celeritas) should be removed from celeritas::detail::EmStandardPhysics.
 */
void SupportedEmStandardPhysics::add_mu_processes(G4ParticleDefinition* p)
{
    auto& ph = *G4PhysicsListHelper::GetPhysicsListHelper();

    if (options_.muon.pair_production)
    {
        ph.RegisterProcess(new G4MuPairProduction(), p);
        CELER_LOG(debug) << "Using muon pair production with "
                            "G4MuPairProductionModel";
    }

    if (options_.muon.ionization)
    {
        ph.RegisterProcess(new G4MuIonisation(), p);
        CELER_LOG(debug) << "Using muon ionization with G4ICRU73QOModel, "
                            "G4BraggModel, and G4MuBetheBlochModel";
    }

    if (options_.muon.bremsstrahlung)
    {
        ph.RegisterProcess(new G4MuBremsstrahlung(), p);
        CELER_LOG(debug) << "Using muon bremsstrahlung with "
                            "G4MuBremsstrahlungModel";
    }

    if (options_.muon.coulomb)
    {
        ph.RegisterProcess(new G4CoulombScattering(), p);
        CELER_LOG(debug) << "Using muon Coulomb scattering with "
                            "G4eCoulombScatteringModel";
    }

    if (options_.muon.msc != MscModelSelection::none)
    {
        auto process = std::make_unique<G4MuMultipleScattering>();
        if (options_.muon.msc == MscModelSelection::wentzelvi)
        {
            process->SetEmModel(new G4WentzelVIModel());
            CELER_LOG(debug) << "Using muon multiple scattering with "
                                "G4WentzelVIModel";
        }
        else if (options_.muon.msc == MscModelSelection::urban)
        {
            process->SetEmModel(new G4UrbanMscModel());
            CELER_LOG(debug) << "Using muon multiple scattering with "
                                "G4UrbanMscModel";
        }
        else
        {
            CELER_VALIDATE(false,
                           << "unsupported muon multiple scattering model "
                              "selection '"
                           << options_.muon.msc << "'");
        }
        ph.RegisterProcess(process.release(), p);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
