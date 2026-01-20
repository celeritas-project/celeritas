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

    // Energy [keV]
    cdf.y = {
        0,
        0.48850540675768084,
        0.8390389347819425,
        1.2521213482687141,
        1.7153033196164724,
        2.253638712180777,
        2.854653691809707,
        3.606073540073316,
        4.470346052913727,
        5.560291219507215,
        6.700556502915258,
        7.953772477101693,
        9.194596305637525,
        10.849180562221111,
        12.353474314071864,
        14.045888515617822,
        15.650634617544647,
        17.38079707555165,
        19.111008546659452,
        19.976130619913615,
        80.0,
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
 * \todo Official tables are implemented directly in C++ in NK Labs Geant4
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
