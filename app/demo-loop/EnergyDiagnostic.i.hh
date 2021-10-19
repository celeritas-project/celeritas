//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnergyDiagnostic.i.hh
//---------------------------------------------------------------------------//

#include "base/CollectionAlgorithms.hh"
#include "base/CollectionBuilder.hh"
#include "base/Span.hh"

namespace demo_loop
{
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

template<MemSpace M>
std::vector<real_type> EnergyDiagnostic<M>::energy_deposition()
{
    // Copy binned energy deposition to host
    std::vector<real_type> edep(energy_by_z_.size());
    celeritas::copy_to_host(energy_by_z_, make_span(edep));
    return edep;
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
