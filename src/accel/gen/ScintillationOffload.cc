//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/gen/ScintillationOffload.cc
//---------------------------------------------------------------------------//
#include "ScintillationOffload.hh"

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
 * Defers physics table preparation to \c G4Scintillation, but also enforces
 * that stacking photons is false afterwards.
 */
void ScintillationOffload::PreparePhysicsTable(
    G4ParticleDefinition const& particle)
{
    G4Scintillation::PreparePhysicsTable(particle);

    // Enforce don't stack photons
    if (this->GetStackPhotons())
    {
        CELER_LOG(warning)
            << "ScintillationOffload requires stacking photons set "
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
 * Geant4. After calling the \c G4Scintillation::PostStepDoIt this function
 * creates a \c GeneratorDistributionData and pushes it to the local offload,
 * which should be \c LocalOpticalGenOffload.
 */
G4VParticleChange*
ScintillationOffload::PostStepDoIt(G4Track const& aTrack, G4Step const& aStep)
{
    CELER_EXPECT(!this->GetStackPhotons());

    auto* result = G4Scintillation::PostStepDoIt(aTrack, aStep);

    if (this->GetNumPhotons() > 0)
    {
        auto data = distribution_from_step(aStep);
        data.type = GeneratorType::scintillation;
        data.num_photons = static_cast<size_type>(this->GetNumPhotons());
        data.continuous_edep_fraction = 1;

        // Push generator distribution for this step to offload
        auto& local = detail::IntegrationSingleton::instance().local_offload();
        auto* gen_offload = dynamic_cast<LocalOpticalGenOffload*>(&local);

        CELER_VALIDATE(gen_offload,
                       << "LocalOpticalGenOffload required for "
                          "ScintillationOffload");

        CELER_LOG_LOCAL(debug)
            << "Offloading " << data.num_photons << " scintillation photons";

        gen_offload->Push(data);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create a generator distribution for the given track and step.
 *
 * Since \c G4Scintillation has a qualified call to its own \c PostStepDoIt
 * method, this override defers to \c ScintillationOffload::PostStepDoIt
 * instead.
 */
G4VParticleChange*
ScintillationOffload::AtRestDoIt(G4Track const& aTrack, G4Step const& aStep)
{
    return this->PostStepDoIt(aTrack, aStep);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
