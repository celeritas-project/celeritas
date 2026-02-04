//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/MucfPhysics.cc
//---------------------------------------------------------------------------//
#include "MucfPhysics.hh"

namespace celeritas
{
namespace inp
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Muon energy CDF data for muon-catalyzed fusion.
 *
 * Data is extracted from https://doi.org/10.1103/PhysRevC.107.034607 .
 */
Grid mucf_muon_energy_cdf()
{
    Grid cdf;
    cdf.interpolation.type = InterpolationType::cubic_spline;

    // Cumulative distribution data [unitless]
    cdf.x = {
        0,
        0.04169381107491854,
        0.08664495114006499,
        0.14332247557003264,
        0.20456026058631915,
        0.2723127035830618,
        0.34136807817589576,
        0.41563517915309456,
        0.48990228013029324,
        0.5667752442996744,
        0.6306188925081434,
        0.6866449511400652,
        0.7309446254071662,
        0.7778501628664496,
        0.8104234527687297,
        0.8403908794788275,
        0.8618892508143323,
        0.8814332247557004,
        0.8970684039087949,
        0.903583061889251,
        1.0,
    };

    // Muon energy [MeV]
    cdf.y = {
        0.,
        0.00048850540675768084,
        0.0008390389347819425,
        0.0012521213482687141,
        0.0017153033196164724,
        0.002253638712180777,
        0.002854653691809707,
        0.003606073540073316,
        0.004470346052913727,
        0.005560291219507215,
        0.006700556502915258,
        0.007953772477101693,
        0.009194596305637525,
        0.010849180562221111,
        0.012353474314071864,
        0.014045888515617822,
        0.015650634617544647,
        0.01738079707555165,
        0.019111008546659452,
        0.019976130619913615,
        0.08,
    };

    CELER_ENSURE(cdf);
    return cdf;
}

//---------------------------------------------------------------------------//
/*!
 * Cycle rate data for muon-catalyzed fusion.
 */
std::vector<MucfCycleRate> mucf_cycle_rates()
{
    //! \todo Implement
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Spin-flip data for muon-catalyzed fusion.
 */
std::vector<MucfAtomSpinFlipRate> mucf_atom_spin_flip_rates()
{
    //! \todo Implement
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Atom transfer data for muon-catalyzed fusion.
 */
std::vector<MucfAtomTransferRate> mucf_atom_transfer_rates()
{
    //! \todo Implement
    return {};
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Initialize with hardcoded values.
 *
 * \note Default data replicated from Acceleron.
 * \todo May need to be updated.
 */
MucfScalars MucfScalars::from_default()
{
    MucfScalars result;
    result.protium = AmuMass{1.007825031898};
    result.deuterium = AmuMass{2.014101777844};
    result.tritium = AmuMass{3.016049281320};
    result.liquid_hydrogen_density = InvCcDensity{4.25e22};
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct hardcoded muon-catalyzed fusion physics data.
 *
 * \todo Official tables are implemented directly in C++ in Acceleron's Geant4
 * implementation. These likely will be loaded from external data files so that
 * muCF experimental data can be loaded at runtime.
 */
MucfPhysics MucfPhysics::from_default()
{
    MucfPhysics result;
    result.scalars = MucfScalars::from_default();
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
