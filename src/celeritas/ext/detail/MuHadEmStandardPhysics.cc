//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/MuHadEmStandardPhysics.cc
//---------------------------------------------------------------------------//
#include "MuHadEmStandardPhysics.hh"

#include <G4BuilderType.hh>
#include <G4EmBuilder.hh>
#include <G4EmParameters.hh>
#include <G4NuclearStopping.hh>
#include <G4PhysicsListHelper.hh>
#include <G4hMultipleScattering.hh>
#include <G4ionIonisation.hh>

#include "G4GenericIon.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with verbosity.
 */
MuHadEmStandardPhysics::MuHadEmStandardPhysics(int verbosity)
{
    G4EmParameters::Instance()->SetVerbose(verbosity);
}

//---------------------------------------------------------------------------//
/*!
 * Build list of particles.
 */
void MuHadEmStandardPhysics::ConstructParticle()
{
    G4EmBuilder::ConstructMinimalEmSet();
}

//---------------------------------------------------------------------------//
/*!
 * Build processes and models.
 */
void MuHadEmStandardPhysics::ConstructProcess()
{
    auto* physics_list = G4PhysicsListHelper::GetPhysicsListHelper();
    G4ParticleDefinition* p = G4GenericIon::GenericIon();

    auto niel_energy_limit = G4EmParameters::Instance()->MaxNIELEnergy();
    G4NuclearStopping* nuclear_stopping = nullptr;
    if (niel_energy_limit > 0)
    {
        // Nuclear stopping is enabled if the energy limit is above zero
        nuclear_stopping = new G4NuclearStopping();
        nuclear_stopping->SetMaxKinEnergy(niel_energy_limit);
    }
    G4hMultipleScattering* ion_msc = new G4hMultipleScattering("ionmsc");
    G4ionIonisation* ion_ionization = new G4ionIonisation();

    physics_list->RegisterProcess(ion_msc, p);
    physics_list->RegisterProcess(ion_ionization, p);
    if (nuclear_stopping)
    {
        physics_list->RegisterProcess(nuclear_stopping, p);
    }

    G4EmBuilder::ConstructCharged(ion_msc, nuclear_stopping);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
