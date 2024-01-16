//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/FastSimulationOffload.cc
//---------------------------------------------------------------------------//
#include "FastSimulationOffload.hh"

#include "corecel/Assert.hh"

#include "ExceptionConverter.hh"
#include "LocalTransporter.hh"
#include "SharedParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a model to be used in all volumes of the problem.
 */
FastSimulationOffload::FastSimulationOffload(G4String const& name,
                                             SharedParams const* params,
                                             LocalTransporter* local)
    : G4VFastSimulationModel(name), params_(params), transport_(local)
{
    CELER_VALIDATE(G4VERSION_NUMBER > 1110,
                   << "the current version of Geant4 (" << G4VERSION_NUMBER
                   << ") is too old to support the fast simulation Flush() "
                      "interface");
    CELER_EXPECT(params);
    CELER_EXPECT(local);
}

//---------------------------------------------------------------------------//
/*!
 * Construct a model for a specific volume of space (an "envelope").
 *
 * The envelope cannot be \c nullptr as this will cause a segmentation fault
 * in the \c G4VFastSimulation base class constructor.
 */
FastSimulationOffload::FastSimulationOffload(G4String const& name,
                                             G4Envelope* region,
                                             SharedParams const* params,
                                             LocalTransporter* local)
    : G4VFastSimulationModel(name, region), params_(params), transport_(local)
{
    CELER_VALIDATE(G4VERSION_NUMBER > 1110,
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
G4bool FastSimulationOffload::IsApplicable(G4ParticleDefinition const& particle)
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
G4bool FastSimulationOffload::ModelTrigger(G4FastTrack const& /*track*/)
{
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Offload the incoming track to Celeritas.
 */
void FastSimulationOffload::DoIt(G4FastTrack const& track, G4FastStep& step)
{
    CELER_EXPECT(track.GetPrimaryTrack());
    CELER_EXPECT(*transport_);

    // Offload this track to Celeritas for transport
    ExceptionConverter call_g4exception{"celer0001", params_};
    CELER_TRY_HANDLE(transport_->Push(*(track.GetPrimaryTrack())),
                     call_g4exception);

    // Kill particle on Geant4 side. Celeritas will take
    // care of energy conservation, so set path, edep to zero.
    step.KillPrimaryTrack();
    step.ProposePrimaryTrackPathLength(0.0);
    step.ProposeTotalEnergyDeposited(0.0);
}

#if G4VERSION_NUMBER >= 1110
//---------------------------------------------------------------------------//
/*!
 * Complete processing of any buffered tracks.
 *
 * Note that this is called in \c G4EventManager::DoProcessing(G4Event*) by
 * \c G4GlobalFastSimulationManager after the main tracking loop has completed.
 * That is done to allow for models that may add "onload" particles back to
 * Geant4.
 */
void FastSimulationOffload::Flush()
{
    ExceptionConverter call_g4exception{"celer0002", params_};
    CELER_TRY_HANDLE(transport_->Flush(), call_g4exception);
}
#endif

}  // namespace celeritas
