//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/MucfPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "corecel/inp/Grid.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion mean cycle rate data.
 *
 * Mean cycle rates are as a function of temperature, with each grid assigned
 * to a muonic molecule and its spin (e.g. \f$ (dt)_\mu, F = 0 \f$).
 *
 * The inverse of the rate provides the time it takes for a muon-catalyzed
 * fusion to happen, from atom formation to fusion. This number is used to
 * update the track time in the stepping loop and allows the competition
 * between fusion and muon decay.
 *
 * Since the cycle rate is per-material, these grids are host-only, with only
 * the final cycle rate (a \c real_type ) for each molecule + spin combination
 * needed in the stepping loop.
 */
struct MucfCycleRate
{
    MucfMuonicMolecule molecule;
    Grid rate;  //!< x = temperature [K], y = mean cycle rate [1/time]
    std::string spin_label;

    //! True if data is assigned
    explicit operator bool() const
    {
        return molecule < MucfMuonicMolecule::size_ && rate
               && !spin_label.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion mean atom transfer data.
 *
 * Atom transfer is not a direct process, encompassing multiple steps:
 * initial_atom --> isotope1 --> isotope2 --> final_atom
 *
 * The transfer rates are as a function of temperature, with a separate grid
 * for each combination of the 4 steps aforementioned
 * (e.g. protium --> deuterium --> tritium --> tritium).
 *
 * Each struct holds one of such combinations, with the full set of
 * combinations being the \c vector<MucfAtomTransferRate> in \c MucfPhysics .
 *
 * \note These grids are host-only, with only the final exchange rate (a
 * \c real_type ) for each combination being needed in the stepping loop. This
 * is because these rates are material dependent, and thus can be cached at
 * model construction.
 */
struct MucfAtomTransferRate
{
    //! \todo Implement
};

//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion mean atom spin flip data.
 *
 * Spin flip rates are as a function of temperature, with each grid/table
 * representing an atom pair combination and its spin (e.g. deuterium-tritium,
 * spin 1). Ordering is important, thus same spin deuterium-tritium and
 * tritium-deuterium have different tables, which leads to a different final
 * spin flip rate for a given material definition.
 *
 * Each struct holds one of such combinations, with the full set of
 * combinations being the \c vector<MucfAtomSpinFlipRate> in \c MucfPhysics .
 *
 * \note These grids are host-only, with only the final spin flip rate per
 * state (which is just a \c real_type ) for each combination being needed in
 * the stepping loop. This is because these rates are material dependent, and
 * thus can be cached at model construction.
 */
struct MucfAtomSpinFlipRate
{
    //! \todo Implement
};

//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion physics options and data import.
 *
 * Minimum requirements for muon-catalyzed fusion:
 * - Muon energy CDF data, required for sampling the outgoing muCF muon, and
 * - Mean cycle rate data for dd, dt, and tt muonic molecules.
 *
 * Muonic atom transfer and muonic atom spin flip are secondary effects and not
 * required for muCF to function.
 */
struct MucfPhysics
{
    template<class T>
    using Vec = std::vector<T>;

    Grid muon_energy_cdf;  //!< CDF for outgoing muCF muon
    Vec<MucfCycleRate> cycle_rates;  //!< Mean cycle rates for muonic molecules
    Vec<MucfAtomTransferRate> atom_transfer;  //!< Muon atom transfer rates
    Vec<MucfAtomSpinFlipRate> atom_spin_flip;  //!< Muon atom spin flip rates

    //! Whether muon-catalyzed fusion physics is enabled
    explicit operator bool() const
    {
        return muon_energy_cdf && !cycle_rates.empty();
    }

    //! Construct hardcoded muon-catalyzed fusion physics data
    static MucfPhysics from_default();
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
