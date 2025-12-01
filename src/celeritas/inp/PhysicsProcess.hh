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
}  // namespace inp
}  // namespace celeritas
