//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnergyDiagnostic.i.hh
//---------------------------------------------------------------------------//

#include "base/Atomics.hh"
#include "base/CollectionAlgorithms.hh"
#include "base/CollectionBuilder.hh"
#include "base/Span.hh"
#include "physics/grid/NonuniformGrid.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
// EnergyDiagnostic implementation
//---------------------------------------------------------------------------//
template<MemSpace M>
EnergyDiagnostic<M>::EnergyDiagnostic(const std::vector<real_type>& z_bounds)
    : Diagnostic<M>()
{
    // Create collection on host and copy to device
    Collection<real_type, Ownership::value, MemSpace::host> z_bounds_host;
    make_builder(&z_bounds_host).insert_back(z_bounds.cbegin(), z_bounds.cend());
    z_bounds_ = z_bounds_host;

    // Resize bin data
    resize(&energy_by_z_, z_bounds_.size() - 1);
}

//---------------------------------------------------------------------------//
/*!
 * Accumulate energy deposition in diagnostic
 */
template<MemSpace M>
void EnergyDiagnostic<M>::end_step(const StateDataRef& states)
{
    // Set up pointers to pass to device
    EnergyBinPointers<M> pointers;
    pointers.z_bounds    = z_bounds_;
    pointers.energy_by_z = energy_by_z_;

    // Invoke kernel for binning energies
    demo_loop::bin_energy(states, pointers);
}

//---------------------------------------------------------------------------//
/*!
 * Get vector of binned energy deposition
 */
template<MemSpace M>
std::vector<real_type> EnergyDiagnostic<M>::energy_deposition()
{
    // Copy binned energy deposition to host
    std::vector<real_type> edep(energy_by_z_.size());
    celeritas::copy_to_host(energy_by_z_, celeritas::make_span(edep));
    return edep;
}

//---------------------------------------------------------------------------//
// EnergyDiagnosticLauncher implementation
//---------------------------------------------------------------------------//
template<MemSpace M>
EnergyDiagnosticLauncher<M>::EnergyDiagnosticLauncher(const StateDataRef& states,
                                                      const Pointers& pointers)
    : states_(states), pointers_(pointers)
{
    CELER_EXPECT(states_);
    CELER_EXPECT(pointers_);
}

//---------------------------------------------------------------------------//
template<MemSpace M>
void EnergyDiagnosticLauncher<M>::operator()(celeritas::ThreadId tid) const
{
    // Create grid from EnergyBinPointers
    celeritas::NonuniformGrid<real_type> grid(pointers_.z_bounds);

    real_type z_pos             = states_.geometry.pos[tid][2];
    real_type energy_deposition = states_.energy_deposition[tid];

    using BinId = celeritas::ItemId<real_type>;
    if (z_pos > grid.front() && z_pos < grid.back())
    {
        auto bin = grid.find(z_pos);
        celeritas::atomic_add(&pointers_.energy_by_z[BinId{bin}],
                              energy_deposition);
    }
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
