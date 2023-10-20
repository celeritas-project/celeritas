//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
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

G4bool FastSimulationOffload::IsApplicable(G4ParticleDefinition const& particle)
{
    CELER_EXPECT(*params_);
 
    return (std::find(params_->OffloadParticles().begin(),
                      params_->OffloadParticles().end(),
                      &particle)
            != params_->OffloadParticles().end());
}

G4bool FastSimulationOffload::ModelTrigger(G4FastTrack const& /*track*/)
{
    // At present we don't offload based on kinematics (or anything else
    // available via G4FastTrack), so always return true
    return true;
}

void FastSimulationOffload::DoIt(G4FastTrack const& track, G4FastStep& step)
{
    // Offload this track to Celeritas for transport
    CELER_EXPECT(track.GetPrimaryTrack());
    CELER_EXPECT(*transport_);
 
    ExceptionConverter call_g4exception{"celer0001", params_};
    CELER_TRY_HANDLE(transport_->Push(*(track.GetPrimaryTrack())),
                     call_g4exception);

    // Kill particle on Geant4 side. Celeritas will take
    // care of energy conservation etc, so set path, edep to zero.
    step.KillPrimaryTrack();
    step.ProposePrimaryTrackPathLength(0.0);
    step.ProposeTotalEnergyDeposited(0.0);
}

#if G4VERSION_NUMBER >= 1110
void FastSimulationOffload::Flush()
{
    ExceptionConverter call_g4exception{"celer0002", params_};
    CELER_TRY_HANDLE(transport_->Flush(), call_g4exception);
}
#endif

}  // namespace celeritas
