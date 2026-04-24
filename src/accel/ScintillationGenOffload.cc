//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/ScintillationGenOffload.cc
//---------------------------------------------------------------------------//
#include "ScintillationGenOffload.hh"

#include <G4Poisson.hh>
#include <G4Step.hh>
#include <G4Track.hh>
#include <Randomize.hh>

#include "corecel/io/Logger.hh"
#include "celeritas/optical/gen/GeneratorData.hh"
#include "accel/LocalOpticalGenOffload.hh"
#include "accel/detail/IntegrationSingleton.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Create a generator distribution for the given track and step.
 *
 * A scintillation distribution is constructed from the given \c G4Step,
 * skipping the initialization of secondary photon tracks in Geant4. The
 * average number of photons is determined either by particle type or through
 * the scintillation yield. If EM saturation is present then it is used for
 * Birk's correction. The resulting distribution is pushed to the local
 * offload, which should be \c LocalOpticalGenOffload.
 */
G4VParticleChange* ScintillationGenOffload::PostStepDoIt(G4Track const& aTrack,
                                                         G4Step const& aStep)
{
    // Calculate number of photons from G4Cerenkov::PostStepDoIt
    aParticleChange.Initialize(aTrack);

    G4double total_energy_deposit = aStep.GetTotalEnergyDeposit();
    if (total_energy_deposit <= 0)
    {
        G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    G4Material const* material = aTrack.GetMaterial();
    G4MaterialPropertiesTable* MPT = material->GetMaterialPropertiesTable();
    if (!MPT)
    {
        return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    G4double mean_num_photons{};
    if (this->GetScintillationByParticleType())
    {
        // mean number determined by particle type
        G4double yield1, yield2, yield3, timeconstant1, timeconstant2,
            timeconstant3;

        mean_num_photons
            = this->GetScintillationYieldByParticleType(aTrack,
                                                        aStep,
                                                        yield1,
                                                        yield2,
                                                        yield3,
                                                        timeconstant1,
                                                        timeconstant2,
                                                        timeconstant3);
    }
    else
    {
        // mean number from linear law [# scintillation photons / MeV]
        mean_num_photons = MPT->GetConstProperty(kSCINTILLATIONYIELD);

        if (this->GetSaturation())
        {
            // Apply Birk's correction if available
            mean_num_photons
                *= this->GetSaturation()->VisibleEnergyDepositionAtAStep(
                    &aStep);
        }
        else
        {
            // Scale by energy deposit
            mean_num_photons *= total_energy_deposit;
        }
    }

    G4int num_photons{};

    if (mean_num_photons > 10)
    {
        G4double sigma = MPT->GetConstProperty(kRESOLUTIONSCALE)
                         * std::sqrt(mean_num_photons);
        num_photons = static_cast<G4int>(
            G4RandGauss::shoot(mean_num_photons, sigma) + 0.5);
    }
    else
    {
        num_photons = static_cast<G4int>(G4Poisson(mean_num_photons));
    }

    if (num_photons > 0)
    {
        // Push generator distribution for this step to offload
        auto& local = detail::IntegrationSingleton::instance().local_offload();
        auto* gen_offload = dynamic_cast<LocalOpticalGenOffload*>(&local);

        CELER_VALIDATE(gen_offload,
                       << "LocalOpticalGenOffload required for "
                          "ScintillationGenOffload");

        CELER_LOG_LOCAL(debug)
            << "Offloading " << num_photons << " scintillation photons";

        gen_offload.Push(aStep,
                         GeneratorType::scintillation,
                         static_cast<size_type>(num_photons));
    }

    // Return particle change
    return pParticleChange;
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
