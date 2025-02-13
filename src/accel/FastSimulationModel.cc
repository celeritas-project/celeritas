//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/FastSimulationModel.cc
//---------------------------------------------------------------------------//
#include "FastSimulationModel.hh"

#include "corecel/Assert.hh"

#include "ExceptionConverter.hh"
#include "LocalTransporter.hh"
#include "SharedParams.hh"

#include "detail/IntegrationSingleton.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct using the FastSimulationIntegration for all regions.
 */
FastSimulationModel::FastSimulationModel(G4Envelope* region)
    : FastSimulationModel(
          "celeritas",
          region,
          &detail::IntegrationSingleton::instance().shared_params(),
          &detail::IntegrationSingleton::local_transporter())
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct without attaching to a region.
 */
FastSimulationModel::FastSimulationModel(G4String const& name,
                                         SharedParams const* params,
                                         LocalTransporter* local)
    : G4VFastSimulationModel(name), params_(params), transport_(local)
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct and build a fast sim manager for the given region.
 *
 * The envelope cannot be \c nullptr as this will cause a segmentation fault
 * in the \c G4VFastSimulation base class constructor.
 */
FastSimulationModel::FastSimulationModel(G4String const& name,
                                         G4Envelope* region,
                                         SharedParams const* params,
                                         LocalTransporter* local)
    : G4VFastSimulationModel(name, region), params_(params), transport_(local)
{
    CELER_VALIDATE(G4VERSION_NUMBER >= 1110,
                   << "the current version of Geant4 (" << G4VERSION_NUMBER
                   << ") is too old to support the fast simulation Flush() "
                      "interface");
    CELER_EXPECT(region);
    CELER_EXPECT(params);
    CELER_EXPECT(local);
}

//---------------------------------------------------------------------------//
/*!
 * Return true if this model can be applied to the input \c
 * G4ParticleDefinition .
 *
 * Purely checks if the particle is one that Celeritas has been setup to
 * handle.
 */
G4bool FastSimulationModel::IsApplicable(G4ParticleDefinition const& particle)
{
    CELER_EXPECT(*params_);

    return (std::find(params_->OffloadParticles().begin(),
                      params_->OffloadParticles().end(),
                      &particle)
            != params_->OffloadParticles().end());
}

//---------------------------------------------------------------------------//
/*!
 * Return true if model can be applied given dynamic particle state in \c
 * G4FastTrack .
 *
 * Always returns true because we only make the decision to offload to
 * Celeritas based on geometric region and particle type.
 */
G4bool FastSimulationModel::ModelTrigger(G4FastTrack const& /*track*/)
{
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Offload the incoming track to Celeritas.
 */
void FastSimulationModel::DoIt(G4FastTrack const& track, G4FastStep& step)
{
    CELER_EXPECT(track.GetPrimaryTrack());

    // Offload this track to Celeritas for transport
    if (*transport_)
    {
        // Offload this track to Celeritas for transport
        CELER_TRY_HANDLE(transport_->Push(*(track.GetPrimaryTrack())),
                         ExceptionConverter("celer.track.push", params_));
    }

    // Kill particle on Geant4 side. Celeritas will take
    // care of energy conservation, so set path, edep to zero.
    step.KillPrimaryTrack();
    step.ProposePrimaryTrackPathLength(0.0);
    step.ProposeTotalEnergyDeposited(0.0);
}

//---------------------------------------------------------------------------//
/*!
 * Complete processing of any buffered tracks.
 *
 * Note that this is called in \c G4EventManager::DoProcessing(G4Event*) by
 * \c G4GlobalFastSimulationManager after the main tracking loop has completed.
 * That is done to allow for models that may add "onload" particles back to
 * Geant4.
 */
void FastSimulationModel::Flush()
{
    if (*transport_)
    {
        CELER_TRY_HANDLE(transport_->Flush(),
                         ExceptionConverter("celer.event.flush", params_));
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
