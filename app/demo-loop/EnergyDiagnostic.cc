//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnergyDiagnostic.cc
//---------------------------------------------------------------------------//
#include "EnergyDiagnostic.hh"

#include "base/Atomics.hh"
#include "base/CollectionBuilder.hh"
#include "base/Macros.hh"
#include "physics/grid/NonuniformGrid.hh"

using namespace celeritas;

namespace demo_loop
{
template<>
void EnergyDiagnostic<MemSpace::host>::end_step(const StateDataRef& states)
{
    using BinId = ItemId<real_type>;

    // Create grid from z_bounds_
    Collection<real_type, Ownership::const_reference, MemSpace::native>
                              z_bounds_ref(z_bounds_);
    NonuniformGrid<real_type> grid(z_bounds_ref);

    for (auto tid : range(ThreadId{states.size()}))
    {
        real_type z_pos             = states.geometry.pos[tid][2];
        real_type energy_deposition = states.energy_deposition[tid];

        if (z_pos > grid.front() && z_pos < grid.back())
        {
            auto bin = grid.find(z_pos);
            atomic_add(&energy_by_z_[BinId{bin}], energy_deposition);
        }
    }
}

} // namespace demo_loop
