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
 * Data is extracted from \citet{kamimura-mucf-2023,
 * https://doi.org/10.1103/PhysRevC.107.034607}.
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
// DD cycle rate data
//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion cycle rate data for \em dd fusion with F = 1/2 state.
 */
MucfCycleRate dd_1_over_2_cycle_data()
{
    MucfCycleRate result;
    result.type = CycleTableType::deuterium_deuterium;
    result.spin_state = units::HalfSpinInt{1};  // F = 1/2

    result.rate.interpolation.type = InterpolationType::cubic_spline;
    // Temperature [K]
    result.rate.x = {
        0.0,
        36.43103676856282,
        49.84753686046977,
        62.30013430590361,
        72.8249264583911,
        94.80254726485413,
        118.65929788178391,
        134.8637502794038,
        174.9275151614521,
        212.16105692405534,
        260.90506935936025,
        325.03565799325077,
        380.6089687138337,
        434.29674495520254,
        486.0836155123639,
        564.7393523983244,
        668.3489596576321,
        764.2819309563899,
        860.2110594538995,
        935.0316806939586,
        998.3392693938174,
    };
    // Mean cycle rate [1/s]
    result.rate.y = {
        0.0,
        0.005464463375221662e6,
        0.020292872692477815e6,
        0.04715981762065269e6,
        0.098103833770665e6,
        0.2721222062034343e6,
        0.5362926959236205e6,
        0.7584940318094278e6,
        1.2388537948623028e6,
        1.6110396634730852e6,
        1.9710284809214524e6,
        2.2225696043387786e6,
        2.3059026708109145e6,
        2.3141137763784285e6,
        2.283275296360035e6,
        2.19792475923249e6,
        2.05207892345204e6,
        1.921378528091838e6,
        1.7996962265613146e6,
        1.7144033314524956e6,
        1.6473195500592115e6,
    };

    CELER_ENSURE(result);
    return result;
}
//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion cycle rate data for \em dd fusion with F = 3/2 state.
 */
MucfCycleRate dd_3_over_2_cycle_data()
{
    MucfCycleRate result;
    result.type = CycleTableType::deuterium_deuterium;
    result.spin_state = units::HalfSpinInt{3};  // F = 3/2

    result.rate.interpolation.type = InterpolationType::cubic_spline;
    // Temperature [K]
    result.rate.x = {
        0.0,
        5.4317995646105715,
        8.023769006655044,
        15.563345056011997,
        21.23403876489806,
        24.19235525928977,
        30.970416194589205,
        36.77176514592074,
        40.6299375993124,
        48.3052926261127,
        54.996250066448454,
        65.48389514020138,
        83.57516295078466,
        104.55813870078731,
        127.50862868996978,
        141.88390769335217,
        157.2358986807023,
        176.44862398894298,
        208.18055529758965,
        248.59273415949275,
        289.00491302139596,
        347.6774429488119,
        437.0821353934164,
        532.1831402218961,
        626.2766906564209,
        748.1620201862349,
        868.0888020131156,
        992.7877025236317,
    };
    // Mean cycle rate [1/s]
    result.rate.y = {
        0.0,
        3.002924371750031,
        3.670220083632165,
        3.976719989803767,
        4.169019528475477,
        3.9765902952616323,
        3.820175795347777,
        3.705860143810432,
        3.6516939388136374,
        3.6395545296699483,
        3.6875501565621276,
        3.8256690797323616,
        4.119986345246231,
        4.378188003927343,
        4.519125621813277,
        4.533939620625851,
        4.506654771061913,
        4.419191653948062,
        4.202281855381299,
        3.8650011112100278,
        3.5277203670387562,
        3.0879607598755188,
        2.527498765500099,
        2.0992156842652023,
        1.785176202044242,
        1.500779127953113,
        1.312603875721246,
        1.1754591026674808,
    };

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
// DT cycle rate data
//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion cycle rate data for \em dt fusion with F = 0 state.
 */
MucfCycleRate dd_0_cycle_data()
{
    MucfCycleRate result;
    result.type = CycleTableType::deuterium_deuterium;
    result.spin_state = units::HalfSpinInt{0};  // F = 0

    result.rate.interpolation.type = InterpolationType::cubic_spline;
    // Temperature [K]
    result.rate.x = {
        9.719,
        61.224,
        113.677,
        149.956,
        203.175,
        301.385,
        407.915,
        491.843,
        609.109,
        731.772,
        859.935,
        1041.085,
        1205.668,
        1370.401,
        1487.748,
    };
    // Mean cycle rate [1/s]
    result.rate.y = {
        0.073e8,
        1.614e8,
        2.294e8,
        2.404e8,
        2.386e8,
        2.195e8,
        2.077e8,
        2.143e8,
        2.447e8,
        2.934e8,
        3.514e8,
        4.286e8,
        4.847e8,
        5.271e8,
        5.502e8,
    };

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion cycle rate data for \em dt fusion with F = 0 state.
 */
MucfCycleRate dt_0_cycle_data()
{
    MucfCycleRate result;
    result.type = CycleTableType::deuterium_tritium;
    result.spin_state = units::HalfSpinInt{0};  // F = 0

    result.rate.interpolation.type = InterpolationType::cubic_spline;
    // Temperature [K]
    result.rate.x = {
        2.800,
        40.599,
        90.979,
        155.357,
        194.476,
        233.474,
        277.910,
        319.385,
        392.354,
        462.421,
        554.546,
        669.030,
        831.184,
        991.254,
        1175.567,
        1349.074,
        1500.362,
    };
    // Mean cycle rate [1/s]
    result.rate.y = {
        0.000e8,
        0.001e8,
        0.020e8,
        0.039e8,
        0.113e8,
        0.296e8,
        0.627e8,
        1.104e8,
        2.224e8,
        3.435e8,
        4.958e8,
        6.518e8,
        8.015e8,
        8.860e8,
        9.303e8,
        9.388e8,
        9.307e8,
    };

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion cycle rate data for \em dt fusion with F = 0 state.
 */
MucfCycleRate hd_0_cycle_data()
{
    MucfCycleRate result;
    result.type = CycleTableType::protium_deuterium;
    result.spin_state = units::HalfSpinInt{0};  // F = 0

    result.rate.interpolation.type = InterpolationType::cubic_spline;
    // Temperature [K]
    result.rate.x = {
        1.400,   65.789,   141.358,  205.626,  237.603,  273.559,  308.014,
        344.996, 409.061,  474.303,  555.983,  628.326,  714.709,  813.915,
        926.103, 1058.264, 1183.899, 1262.237, 1368.676, 1500.476,
    };
    // Mean cycle rate [1/s]
    result.rate.y = {
        0.000e8,  0.010e8,  0.039e8,  0.159e8,  0.361e8,  0.765e8,  1.260e8,
        2.003e8,  3.581e8,  5.361e8,  7.470e8,  9.158e8,  10.810e8, 12.260e8,
        13.361e8, 14.124e8, 14.456e8, 14.512e8, 14.476e8, 14.295e8,
    };

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion cycle rate data for \em dt fusion with F = 1 state.
 */
MucfCycleRate dd_1_cycle_data()
{
    MucfCycleRate result;
    result.type = CycleTableType::deuterium_deuterium;
    result.spin_state = units::HalfSpinInt{2};  // F = 1

    result.rate.interpolation.type = InterpolationType::cubic_spline;
    // Temperature [K]
    result.rate.x = {
        0.0,
        40.97238,
        86.18093,
        154.00251,
        216.16937,
        309.43132,
        391.40911,
        484.73130,
        592.21748,
        733.66109,
        900.53147,
        1070.18645,
        1251.11005,
        1383.94917,
        1501.23102,
    };
    // Mean cycle rate [1/s]
    result.rate.y = {
        0.0e8,
        0.02590e8,
        0.03788e8,
        0.11774e8,
        0.17032e8,
        0.33172e8,
        0.61735e8,
        1.20516e8,
        2.05375e8,
        3.27240e8,
        4.47630e8,
        5.39123e8,
        6.07188e8,
        6.38302e8,
        6.57099e8,
    };

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion cycle rate data for \em dt fusion with F = 1 state.
 */
MucfCycleRate dt_1_cycle_data()
{
    MucfCycleRate result;
    result.type = CycleTableType::deuterium_tritium;
    result.spin_state = units::HalfSpinInt{2};  // F = 1

    result.rate.interpolation.type = InterpolationType::cubic_spline;
    // Temperature [K]
    result.rate.x = {
        0.0,
        42.3405,
        90.3261,
        149.6118,
        193.4004,
        257.0266,
        312.1948,
        377.2730,
        445.1485,
        527.1292,
        630.2549,
        731.9188,
        825.0831,
        935.1662,
        1079.0999,
        1268.1724,
        1502.3866,
    };
    // Mean cycle rate [1/s]
    result.rate.y = {
        0.0,
        0.0254e8,
        0.0505e8,
        0.2126e8,
        0.7603e8,
        2.4341e8,
        4.2458e8,
        6.4969e8,
        8.3905e8,
        10.1734e8,
        11.6117e8,
        12.3352e8,
        12.6055e8,
        12.6414e8,
        12.3869e8,
        11.8140e8,
        10.9640e8,
    };

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion cycle rate data for \em dt fusion with F = 1 state.
 */
MucfCycleRate hd_1_cycle_data()
{
    MucfCycleRate result;
    result.type = CycleTableType::protium_deuterium;
    result.spin_state = units::HalfSpinInt{2};  // F = 1

    result.rate.interpolation.type = InterpolationType::cubic_spline;
    // Temperature [K]
    result.rate.x = {
        0.0,        57.92486,   108.78618,  151.19847, 182.34991, 197.94409,
        227.75862,  274.69662,  320.26078,  380.06082, 442.66685, 515.12296,
        595.97369,  690.86987,  795.54230,  883.16703, 976.39400, 1089.35418,
        1261.56836, 1405.53035, 1505.73924,
    };
    // Mean cycle rate [1/s]
    result.rate.y = {
        0.0,        0.02523e8,  0.05075e8,  0.26916e8,  0.77688e8,  1.16141e8,
        2.20563e8,  4.45963e8,  6.98879e8,  10.28768e8, 13.44890e8, 16.33464e8,
        18.64233e8, 20.37175e8, 21.30299e8, 21.56089e8, 21.47469e8, 21.07136e8,
        20.10173e8, 19.14696e8, 18.48278e8,
    };

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
// TT cycle rate data
//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion cycle rate data for \em tt fusion with F = 1/2 state.
 *
 * The tt fusion cycle rate data is currently a constant value at 2.8e6 1/s.
 */
MucfCycleRate tt_1_over_2_cycle_data()
{
    MucfCycleRate result;
    result.type = CycleTableType::tritium_tritium;
    result.spin_state = units::HalfSpinInt{1};  // F = 1/2

    result.rate.interpolation.type = InterpolationType::linear;
    // Temperature [K]
    result.rate.x = {
        0.0,
        std::numeric_limits<double>::max(),
    };
    // Mean cycle rate [1/s]
    result.rate.y = {2.8e6, 2.8e6};

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Cycle rate data for muon-catalyzed fusion.
 *
 * Data is extracted from \citet{faifman-mucfformation-1996,
 * https://doi.org/10.1007/BF02227621}.
 *
 * \todo Use native units.
 */
std::vector<MucfCycleRate> mucf_cycle_rates()
{
    std::vector<MucfCycleRate> result{
        // DD fusion cycle rates
        dd_1_over_2_cycle_data(),
        dd_3_over_2_cycle_data(),
        // DT fusion cycle rates
        hd_0_cycle_data(),
        dd_0_cycle_data(),
        dt_0_cycle_data(),
        hd_1_cycle_data(),
        dd_1_cycle_data(),
        dt_1_cycle_data(),
        // TT fusion cycle rates
        tt_1_over_2_cycle_data(),
    };
    return result;
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

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
