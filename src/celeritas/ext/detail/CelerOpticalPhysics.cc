//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/CelerOpticalPhysics.cc
//---------------------------------------------------------------------------//
#include "CelerOpticalPhysics.hh"

#include <memory>
#include <G4Cerenkov.hh>
#include <G4EmSaturation.hh>
#include <G4LossTableManager.hh>
#include <G4OpAbsorption.hh>
#include <G4OpBoundaryProcess.hh>
#include <G4OpMieHG.hh>
#include <G4OpRayleigh.hh>
#include <G4OpWLS.hh>
#include <G4ParticleDefinition.hh>
#include <G4ProcessManager.hh>
#include <G4Scintillation.hh>
#include <G4Version.hh>
#if G4VERSION_NUMBER >= 1070
#    include <G4OpWLS2.hh>
#    include <G4OpticalParameters.hh>
#endif

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with physics options.
 */
CelerOpticalPhysics::CelerOpticalPhysics(Options const& options)
    : options_(options)
{
#if G4VERSION_NUMBER >= 1070
    // Use of G4OpticalParameters only from Geant4 10.7
    auto* params = G4OpticalParameters::Instance();
    CELER_ASSERT(params);

    auto activate_process = [params](G4OpticalProcessIndex i, bool flag) {
        params->SetProcessActivation(G4OpticalProcessName(i), flag);
    };

    activate_process(kCerenkov, options_.cerenkov_radiation);
    activate_process(kScintillation, options_.scintillation);
    activate_process(kAbsorption, options_.absorption);
    activate_process(kRayleigh, options_.rayleigh_scattering);
    activate_process(kMieHG, options_.mie_scattering);
    activate_process(kBoundary, options_.boundary);
    activate_process(
        kWLS, options_.wavelength_shifting != WLSTimeProfileSelection::none);
    activate_process(
        kWLS2, options_.wavelength_shifting2 != WLSTimeProfileSelection::none);

    // Cerenkov
    params->SetCerenkovStackPhotons(options_.cerenkov_stack_photons);
    params->SetCerenkovTrackSecondariesFirst(
        options_.cerenkov_track_secondaries_first);
    params->SetCerenkovMaxPhotonsPerStep(options_.cerenkov_max_photons);
    params->SetCerenkovMaxBetaChange(options_.cerenkov_max_beta_change);

    // Scintillation
    params->SetScintStackPhotons(options_.scint_stack_photons);
    params->SetScintTrackSecondariesFirst(
        options_.scint_track_secondaries_first);
    params->SetScintByParticleType(options_.scint_by_particle_type);
    params->SetScintFiniteRiseTime(options_.scint_finite_rise_time);
    params->SetScintTrackInfo(options_.scint_track_info);

    // WLS
    params->SetWLSTimeProfile(to_cstring(options_.wavelength_shifting));

    // WLS2
    params->SetWLS2TimeProfile(to_cstring(options_.wavelength_shifting2));

    // boundary
    params->SetBoundaryInvokeSD(options_.invoke_sd);

    // Only set a global verbosity with same level for all optical processes
    params->SetVerboseLevel(options_.verbose);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Build list of available particles.
 */
void CelerOpticalPhysics::ConstructParticle()
{
    // Eventually nothing to do here as Celeritas OpPhys won't generate
    // G4OpticalPhotons
    G4OpticalPhoton::OpticalPhotonDefinition();
}

//---------------------------------------------------------------------------//
/*!
 * Build list of available processes and models.
 */
void CelerOpticalPhysics::ConstructProcess()
{
    auto* process_manager
        = G4OpticalPhoton::OpticalPhoton()->GetProcessManager();
    CELER_ASSERT(process_manager);

    // Add Optical Processes
    // TODO: Celeritas will eventually implement these directly (no
    // G4OpticalPhotons) so how to set up on "Celeritas-side"
    auto absorption = std::make_unique<G4OpAbsorption>();
    if (process_is_active("OpAbsorption"))
    {
        process_manager->AddDiscreteProcess(absorption.release());
        CELER_LOG(debug) << "Loaded Optical absorption with G4OpAbsorption "
                            "process";
    }

    auto rayleigh = std::make_unique<G4OpRayleigh>();
    if (process_is_active("OpRayleigh"))
    {
        process_manager->AddDiscreteProcess(rayleigh.release());
        CELER_LOG(debug)
            << "Loaded Optical Rayleigh scattering with G4OpRayleigh "
               "process";
    }

    auto mie = std::make_unique<G4OpMieHG>();
    if (process_is_active("OpMieHG"))
    {
        process_manager->AddDiscreteProcess(mie.release());
        CELER_LOG(debug) << "Loaded Optical Mie (Henyey-Greenstein phase "
                            "function) scattering with G4OpMieHG "
                            "process";
    }

    // NB: boundary is also used later on in loop over particles,
    // though it's only ever applicable to G4OpticalPhotons
    auto boundary = std::make_unique<G4OpBoundaryProcess>();
#if G4VERSION_NUMBER < 1070
    boundary->SetInvokeSD(options_.invoke_sd);
#endif
    if (process_is_active("OpBoundary"))
    {
        process_manager->AddDiscreteProcess(boundary.get());
        CELER_LOG(debug)
            << "Loaded Optical boundary process with G4OpBoundaryProcess "
               "process";
    }

    auto wls = std::make_unique<G4OpWLS>();
#if G4VERSION_NUMBER < 1070
    wls->UseTimeProfile(to_cstring(options_.wavelength_shifting);
#endif
    if (process_is_active("OpWLS"))
    {
        process_manager->AddDiscreteProcess(wls.release());
        CELER_LOG(debug) << "Loaded Optical wavelength shifting with G4OpWLS "
                            "process";
    }

#if G4VERSION_NUMBER >= 1070
    auto wls2 = std::make_unique<G4OpWLS2>();
    if (process_is_active("OpWLS2"))
    {
        process_manager->AddDiscreteProcess(wls2.release());
        // I need to check how this differs from G4OpWLS...
        CELER_LOG(debug)
            << "Loaded Optical wavelength shifting V2 with G4OpWLS2 "
               "process";
    }
#endif

    // Add photon-generating processes to all particles they apply to
    // TODO: Eventually replace with Celeritas step collector processes
    auto scint = std::make_unique<G4Scintillation>();
#if G4VERSION_NUMBER < 1070
    scint->SetStackPhotons(options_.scint_stack_photons);
    scint->SetTrackSecondariesFirst(options_.scint_track_secondaries_first);
    scint->SetScintillationByParticleType(options_.scint_by_particle_type);
    scint->SetFiniteRiseTime(options_.scint_finite_rise_time);
    scint->SetScintillationTrackInfo(options_.scint_track_info);
    // These two are not in 10.7 and newer, but defaults should be sufficient
    // for now
    // scint->SetScintillationYieldFactor(fYieldFactor);
    // scint->SetScintillationExcitationRatio(fExcitationRatio);
#endif
    scint->AddSaturation(G4LossTableManager::Instance()->EmSaturation());

    auto cerenkov = std::make_unique<G4Cerenkov>();
#if G4VERSION_NUMBER < 1070
    cerenkov->SetStackPhotons(options_.cerenkov_stack_photons);
    cerenkov->SetTrackSecondariesFirst(options_.cerenkov_track_secondaries_first);
    cerenkov->SetMaxNumPhotonsPerStep(options_.cerenkov_max_photons);
    cerenkov->SetMaxBetaChangePerStep(options_.cerenkov_max_beta_change);
#endif

    auto particle_iterator = GetParticleIterator();
    particle_iterator->reset();

    while ((*particle_iterator)())
    {
        G4ParticleDefinition* p = particle_iterator->value();
        process_manager = p->GetProcessManager();
        CELER_ASSERT(process_manager);

        if (cerenkov->IsApplicable(*p) && process_is_active("Cerenkov"))
        {
            process_manager->AddProcess(cerenkov.get());
            process_manager->SetProcessOrdering(cerenkov.get(), idxPostStep);
            CELER_LOG(debug) << "Loaded Optical Cerenkov with G4Cerenkov "
                                "process for particle "
                             << p->GetParticleName();
        }
        if (scint->IsApplicable(*p) && process_is_active("Scintillation"))
        {
            process_manager->AddProcess(scint.get());
            process_manager->SetProcessOrderingToLast(scint.get(), idxAtRest);
            process_manager->SetProcessOrderingToLast(scint.get(), idxPostStep);
            CELER_LOG(debug)
                << "Loaded Optical Scintillation with G4Scintillation "
                   "process for particle "
                << p->GetParticleName();
        }
        if (boundary->IsApplicable(*p) && process_is_active("OpBoundary"))
        {
            process_manager->SetProcessOrderingToLast(boundary.get(),
                                                      idxPostStep);
        }
    }
    boundary.release();
    cerenkov.release();
    scint.release();
}

//---------------------------------------------------------------------------//
// PRIVATE
//---------------------------------------------------------------------------//
/*!
 * Return true if a given process is active
 *
 * Use `G4OpticalParameters` when available, otherwise use hardcoded checks.
 */
bool CelerOpticalPhysics::process_is_active(
    [[maybe_unused]] std::string const& process_name)
{
#if G4VERSION_NUMBER >= 1070
    auto* params = G4OpticalParameters::Instance();
    CELER_ASSERT(params);
    return params->GetProcessActivation(process_name);
#else
    if (process_name == "Cerenkov")
        return options_.cerenkov_radiation;
    else if (process_name == "Scintillation")
        return options_.scintillation;
    else if (process_name == "OpAbsorption")
        return options_.absorption;
    else if (process_name == "OpBoundary")
        return options_.boundary;
    else if (process_name == "OpMieHG")
        return options_.mie_scattering;
    else if (process_name == "OpRayleigh")
        return options_.rayleigh_scattering;
    else if (process_name == "OpWLS")
        return options_.wavelength_shifting != WLSTimeProfileSelection::none;

    CELER_UNREACHABLE();
    return false;
#endif
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
