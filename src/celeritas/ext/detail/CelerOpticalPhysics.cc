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
namespace
{
//---------------------------------------------------------------------------//
bool process_is_active(std::string const& process_name)
{
#if G4VERSION_NUMBER >= 1070
    auto* params = G4OpticalParameters::Instance();
    CELER_ASSERT(params);
    return params->GetProcessActivation(process_name);
#else
    return true;
#endif
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with physics options.
 */
CelerOpticalPhysics::CelerOpticalPhysics(Options const& options)
    : options_(options)
{
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
    if (process_is_active("OpBoundary"))
    {
        process_manager->AddDiscreteProcess(boundary.get());
        CELER_LOG(debug)
            << "Loaded Optical boundary process with G4OpBoundaryProcess "
               "process";
    }

    auto wls = std::make_unique<G4OpWLS>();
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
    scint->AddSaturation(G4LossTableManager::Instance()->EmSaturation());

    auto cerenkov = std::make_unique<G4Cerenkov>();

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

}  // namespace detail
}  // namespace celeritas
