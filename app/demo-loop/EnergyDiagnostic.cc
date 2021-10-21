//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnergyDiagnostic.cc
//---------------------------------------------------------------------------//
#include "EnergyDiagnostic.hh"

#include "base/Atomics.hh"
#include "base/Range.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
void bin_energy(const StateDataRefHost& states, PointersHost& pointers)
{
    EnergyDiagnosticLauncher<MemSpace::host> launch(states, pointers);
    for (auto tid : range(ThreadId{states.size()}))
    {
        launch(tid);
    }
}

} // namespace demo_loop
