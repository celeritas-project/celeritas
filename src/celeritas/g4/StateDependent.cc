//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/StateDependent.cc
//---------------------------------------------------------------------------//
#include "StateDependent.hh"

#include <G4StateManager.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
G4bool StateDependent::Notify(G4ApplicationState state)
{
    G4StateManager* sm = G4StateManager::GetStateManager();
    CELER_ASSERT(sm);
    G4ApplicationState prev = sm->GetPreviousState();
    this->cb_(local_stream_, prev, state);
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
