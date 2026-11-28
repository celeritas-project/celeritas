//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/OnloadProcessor.cc
//---------------------------------------------------------------------------//
#include "OnloadProcessor.hh"

#include <iostream>
#include <G4Gamma.hh>
#include <G4HadronicProcessType.hh>
#include <G4Navigator.hh>
#include <G4TransportationManager.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/Assert.hh"
#include "celeritas/ext/HadronicInteractor.hh"
#include "celeritas/user/OnloadData.hh"
#include "celeritas/user/OnloadSteps.hh"

// Test
#include <G4LogicalVolume.hh>
#include <G4ThreeVector.hh>
#include <G4TouchableHistoryHandle.hh>

#include "StepProcessor.t.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from particles.
 *
 * The base class creates the shared step, track reconstruction, and (if a
 * device is active) the event for copying device steps.
 */
OnloadProcessor::OnloadProcessor(VecParticle const& particles)
    : OnloadProcessorBase(particles)
{
    G4VPhysicalVolume* world
        = G4TransportationManager::GetTransportationManager()
              ->GetNavigatorForTracking()
              ->GetWorldVolume();

    // Create a dedicated navigator
    navigator = std::make_shared<G4Navigator>();
    navigator->SetWorldVolume(world);

    interactor = std::make_shared<HadronicInteractor>(
        *G4Gamma::Gamma(), G4HadronicProcessType::fHadronInelastic);
}

//---------------------------------------------------------------------------//
/*!
 * Generate and call a single onloaded step process.
 */
void OnloadProcessor::operator()(OnloadStepOutput const& out, size_type i) const
{
    CELER_EXPECT(i < out.size());
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//
template class StepProcessor<OnloadProcessor,
                             OnloadStepOutput,
                             HostRef<OnloadStateData>,
                             DeviceRef<OnloadStateData>>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
