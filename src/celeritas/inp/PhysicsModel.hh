//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/PhysicsModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>

#include "celeritas/io/ImportLivermorePE.hh"
#include "celeritas/phys/AtomicNumber.hh"

#include "Grid.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Seltzer-Berger bremsstrahlung model.
 */
struct SeltzerBergerModel
{
    //! Differential cross sections [(log MeV, unitless) -> millibarn]
    std::map<AtomicNumber, inp::TwodGrid> atomic_xs;

    //! TODO: microscopic elemental xs tables

    //! Whether model has data and is to be used
    explicit operator bool() const { return !atomic_xs.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Relativistic bremsstrahlung model.
 */
struct RelBremsModel
{
    //! Account for LPM effect at very high energies
    bool enable_lpm{true};

    //! Whether model has data and is to be used
    explicit operator bool() const { return false; }
};

//---------------------------------------------------------------------------//
/*!
 * Muon bremsstrahlung model.
 */
struct MuBremsModel
{
    //! Whether model has data and is to be used
    explicit operator bool() const { return false; }
};

//---------------------------------------------------------------------------//
// PAIR PRODUCTION MODELS
//---------------------------------------------------------------------------//
/*!
 * Bethe-Heitler relativistic pair production from gammas.
 */
struct BetheHeitlerProductionModel
{
    //! Whether model has data and is to be used
    explicit operator bool() const { return false; }
};

//---------------------------------------------------------------------------//
/*!
 * Sampling table for electron-positron pair production by muons.
 *
 * This 3-dimensional table is used to sample the energy transfer to the
 * electron-positron pair, \f$ \epsilon_p \f$. The outer grid stores the atomic
 * number using 5 equally spaced points in \f$ \log Z \f$; the x grid stores
 * the logarithm of the incident muon energy \f$ T \f$ using equal spacing
 * in \f$ \log T \f$; the y grid stores the ratio \f$ \log \epsilon_p / T \f$.
 * The values are the unnormalized CDF.
 *
 * \todo move directly into MuPairProductionModel?
 */
struct MuPairProductionEnergyTransferTable
{
    //! Z grid for sampling table
    std::vector<AtomicNumber> atomic_number;
    //! Sampling tables for energy transfer at Z [(log MeV, ratio) -> cdf]
    std::vector<inp::TwodGrid> grids;

    //! True if data is assigned
    explicit operator bool() const
    {
        return !atomic_number.empty() && grids.size() == atomic_number.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Pair production from muons.
 */
struct MuPairProductionModel
{
    //! Grid for sampling the energy of the electron-positron
    MuPairProductionEnergyTransferTable muppet_table;

    //! True if data is assigned
    explicit operator bool() const { return static_cast<bool>(muppet_table); }
};

//---------------------------------------------------------------------------//
// PHOTOELECTRIC EFFECT
//---------------------------------------------------------------------------//
// TODO: port these
using LivermoreXs = ::celeritas::ImportLivermorePE;

//---------------------------------------------------------------------------//
/*!
 * Livermore photoelectric effect model.
 */
struct LivermorePhotoModel
{
    //! Tabulated microsopic cross sections [MeV -> b]
    std::map<AtomicNumber, LivermoreXs> atomic_xs;

    //! Whether model has data and is to be used
    explicit operator bool() const { return !atomic_xs.empty(); }
};

//---------------------------------------------------------------------------//
// ALIASES
//---------------------------------------------------------------------------//
//!@{
//! \name Model aliases
//! \todo rename `em/model` to match

using BetheHeitlerModel = BetheHeitlerProductionModel;
using MuBremsstrahlungModel = MuBremsModel;
using RelativisticBremModel = RelBremsModel;

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
