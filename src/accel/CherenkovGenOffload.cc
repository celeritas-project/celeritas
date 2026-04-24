//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/CherenkovGenOffload.cc
//---------------------------------------------------------------------------//
#include "CherenkovGenOffload.hh"

#include <G4Poisson.hh>
#include <G4Step.hh>
#include <G4Track.hh>

#include "celeritas/optical/gen/GeneratorData.hh"
#include "accel/LocalOpticalGenOffload.hh"
#include "accel/detail/IntegrationSingleton.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 */
G4VParticleChange*
CherenkovGenOffload::PostStepDoIt(G4Track const& aTrack, G4Step const& aStep)
{
    // Calculate number of photons from G4Cerenkov::PostStepDoIt
    aParticleChange.Initialize(aTrack);

    G4Material const* material = aTrack.GetMaterial();
    G4MaterialPropertiesTable* MPT = material->GetMaterialPropertiesTable();
    if (!MPT)
    {
        return pParticleChange;
    }

    G4MaterialPropertyVector* r_index = MPT->GetProperty(kRINDEX);
    if (!r_index)
    {
        return pParticleChange;
    }

    G4double charge
        = aTrack.GetDynamicParticle()->GetDefinition()->GetPDGCharge();
    G4double beta = (aStep.GetPreStepPoint()->GetBeta()
                     + aStep.GetPostStepPoint()->GetBeta())
                    * 0.5;

    size_type num_photons = static_cast<size_type>(G4Poisson(
        this->GetAverageNumberOfPhotons(charge, beta, material, r_index)));

    if (num_photons > 0)
    {
        // Push generator distribution for this step to offload
        auto& local = detail::IntegrationSingleton::instance().local_offload();
        auto& gen_offload = dynamic_cast<LocalOpticalGenOffload&>(local);
        gen_offload.Push(aStep, GeneratorType::cherenkov, num_photons);
    }

    // Return particle change
    return pParticleChange;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
