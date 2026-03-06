//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/HadronicInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "G4HadronicProcessType.hh"
#include "G4String.hh"

class G4HadronicProcess;
class G4VParticleChange;
class G4ParticleDefinition;
class G4Step;
class G4Track;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interface for invoking the PostStepDoIt action of Geant4 hadronic processes
 * for a given particle definition and hadronic process type.
 *
 * This class is used to sample final states of hadronic process type
 * that are not yet implemented on GPU. In such cases, the interaction is
 * processed on the CPU using Geant4 with reconstructed G4Track and G4Step
 * objects.
 */
class HadronicInteractor
{
  public:
    // Construct with particle definition and hadronic process type
    HadronicInteractor(G4ParticleDefinition const& particle,
                       G4HadronicProcessType type);

    // Process PostStepDoIt
    G4VParticleChange* PostStepDoIt(G4Track const& track, G4Step const& step);

    // Return the underlying process name
    G4String GetProcessName() const;

  private:
    //// DATA ////

    // Particle definition for which the process is valid
    G4ParticleDefinition const& particle_;

    // Hadronic process obtained from Geant4 process store
    G4HadronicProcess* process_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
