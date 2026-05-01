//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/gen/CherenkovOffload.cc
//---------------------------------------------------------------------------//
#include "CherenkovOffload.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/g4/GeantOffloadUtils.hh"
#include "celeritas/optical/gen/GeneratorData.hh"
#include "accel/LocalOpticalGenOffload.hh"
#include "accel/detail/IntegrationSingleton.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Prepare physics table for particle and enforce photon stacking.
 *
 * Defers physics table preparation to \c G4Cerenkov, but also enforces that
 * stacking photons is false afterwards.
 */
void CherenkovOffload::PreparePhysicsTable(G4ParticleDefinition const& particle)
{
    G4Cerenkov::PreparePhysicsTable(particle);

    // Enforce don't stack photons
    if (this->GetStackPhotons())
    {
        CELER_LOG(warning) << "CherenkovOffload requires stacking photons set "
                              "to false since it sends optical photon tracks "
                              "directly to Celeritas.";
        this->SetStackPhotons(false);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create a generator distribution for the given track and step.
 *
 * Stacking photons should be disabled so that photons are not duplicated in
 * Geant4. After calling the \c G4Cerenkov::PostStepDoIt this function creates
 * a \c GeneratorDistributionData and pushes it to the local offload, which
 * should be \c LocalOpticalGenOffload.
 */
G4VParticleChange*
CherenkovOffload::PostStepDoIt(G4Track const& aTrack, G4Step const& aStep)
{
    CELER_EXPECT(!this->GetStackPhotons());

    auto* result = G4Cerenkov::PostStepDoIt(aTrack, aStep);

    if (this->GetNumPhotons() > 0)
    {
        auto data = distribution_from_step(aStep);
        data.type = GeneratorType::cherenkov;
        data.num_photons = static_cast<size_type>(this->GetNumPhotons());

        // Push generator distribution for this step to offload
        auto& local = detail::IntegrationSingleton::instance().local_offload();
        auto* gen_offload = dynamic_cast<LocalOpticalGenOffload*>(&local);

        CELER_VALIDATE(gen_offload,
                       << "LocalOpticalGenOffload required for "
                          "CherenkovOffload");

        CELER_LOG_LOCAL(debug)
            << "Offloading " << data.num_photons << " Cherenkov photons";

        gen_offload->Push(data);
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
