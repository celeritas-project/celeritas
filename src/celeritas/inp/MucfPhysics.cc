//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/MucfPhysics.cc
//---------------------------------------------------------------------------//
#include "MucfPhysics.hh"

#include "MucfPhysicsData.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Construct hardcoded muon-catalyzed fusion physics data.
 *
 * \todo Official tables are implemented directly in C++ in NK Labs Geant4
 * implementation. These likely will be loaded from external data files so that
 * muCF experimental data can be loaded at runtime.
 */
MucfPhysics MucfPhysics::from_default()
{
    MucfPhysics result;
    result.muon_energy_cdf = mucf_muon_energy_cdf();
    result.cycle_rates = mucf_cycle_rates();
    result.atom_transfer = mucf_atom_transfer_rates();
    result.atom_spin_flip = mucf_atom_spin_flip_rates();

    // Temporary test dummy data to verify correct import
    {
        MucfCycleRate dt_cycle;
        dt_cycle.molecule = MucfMuonicMolecule::deuterium_tritium;

        dt_cycle.spin_label = "F=0";
        dt_cycle.rate = Grid::from_constant(2.0);
        result.cycle_rates.push_back(dt_cycle);

        dt_cycle.spin_label = "F=1";
        dt_cycle.rate = Grid::from_constant(3.0);
        result.cycle_rates.push_back(dt_cycle);
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
