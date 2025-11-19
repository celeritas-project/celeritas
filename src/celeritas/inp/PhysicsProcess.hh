//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/PhysicsProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>

#include "celeritas/io/ImportAtomicRelaxation.hh"
#include "celeritas/phys/AtomicNumber.hh"

#include "PhysicsModel.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Construct a physics process for bremsstrahlung.
 */
struct BremsstrahlungProcess
{
    //! Lower-energy electron/positron
    SeltzerBergerModel sb;
    //! High-energy electron/positron
    RelBremsModel rel;
    //! Muon (-/+)
    MuBremsModel mu;

    //! TODO: macroscopic xs tables

    //! Whether process has data and is to be used
    explicit operator bool() const { return sb || rel || mu; }
};

//---------------------------------------------------------------------------//
/*!
 * Construct a physics process for electron/positron pair production.
 */
struct PairProductionProcess
{
    //! Pair production from gammas
    BetheHeitlerProductionModel bethe_heitler;
    //! Pair production from muons
    MuPairProductionModel mu;

    //! Whether process has data and is to be used
    explicit operator bool() const { return bethe_heitler || mu; }
};

//---------------------------------------------------------------------------//
/*!
 * Construct a physics process for photoelectric effect.
 */
struct PhotoelectricProcess
{
    LivermorePhotoModel livermore;

    //! Whether process has data and is to be used
    explicit operator bool() const { return static_cast<bool>(livermore); }
};

//---------------------------------------------------------------------------//
/*!
 * Emit fluorescence photons/auger electrons from atomic de-excitation.
 *
 * \todo Since multiple processes can cause the loss of a bound electron, we
 * should have a separate "deexcitation" process that manages this efficiently.
 * (Or perhaps a "generator" class to emit many simultaneously.)
 */
struct AtomicRelaxation
{
    //! Differential cross sections [(log MeV, unitless) -> millibarn]
    std::map<AtomicNumber, ImportAtomicRelaxation> atomic_xs;

    //! True if data is assigned
    explicit operator bool() const { return !atomic_xs.empty(); }
};

//---------------------------------------------------------------------------//
//!@{
//! \name Process aliases
//! \todo rename `em/model` to match, merge muon and electron processes

using GammaConversionProcess = PairProductionProcess;
using MuPairProductionProcess = PairProductionProcess;

//!@}

//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion mean cycle rate data.
 *
 * Mean cycle rates are as a function of temperature, with each grid assigned
 * to a muonic molecule and its spin (e.g. \f$ (dt)_\mu, F = 0 \f$).
 */
struct CycleRateData
{
    MuonicMolecule molecule;
    Grid grid;
    std::string spin_label;

    //! True if data is assigned
    explicit operator bool() const
    {
        return molecule < MuonicMolecule::size_ && grid && !spin_label.empty();
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
 * for each combination of the 4 steps below
 * (e.g. protium --> deuterium --> tritium --> tritium).
 *
 * \note These grids are host-only, with only the final exchange rate (a
 * \c real_type ) for each combination being needed in the stepping loop. This
 * is because these rates are material dependent, and thus can be cached at
 * model construction.
 */
struct AtomTransferRateData
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
 * tritium-deuterium have different tables.
 *
 * \note These grids are host-only, with only the final spin flip rate per
 * state (which is just a \c real_type ) for each combination being needed in
 * the stepping loop. This is because these rates are material dependent, and
 * thus can be cached at model construction.
 */
struct AtomSpinFlipRateData
{
    //! \todo Implement
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
