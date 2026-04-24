//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/gen/G4CherenkovOffload.cc
//---------------------------------------------------------------------------//
#include "G4CherenkovOffload.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/optical/gen/GeneratorData.hh"
#include "accel/LocalOpticalGenOffload.hh"
#include "accel/detail/IntegrationSingleton.hh"

#include "G4OffloadUtils.hh"

namespace celeritas
{
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
G4CherenkovOffload::PostStepDoIt(G4Track const& aTrack, G4Step const& aStep)
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
                          "G4CherenkovOffload");

        CELER_LOG_LOCAL(debug)
            << "Offloading " << data.num_photons << " Cherenkov photons";

        gen_offload->Push(data);
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
