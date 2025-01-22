//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/PhysicsModel.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Seltzer-Berger bremsstrahlung model.
 *
 * \todo Move \c sb_data from celeritas::ImportData here.
 */
struct SeltzerBergerModel
{
};

//---------------------------------------------------------------------------//
/*!
 * Relativistic bremsstrahlung model.
 */
struct RelBremsModel
{
    //! Account for LPM effect at very high energies
    bool enable_lpm{true};
};

//---------------------------------------------------------------------------//
/*!
 * Muon bremsstrahlung model.
 */
struct MuBremsModel
{
};

//---------------------------------------------------------------------------//
// PAIR PRODUCTION MODELS
//---------------------------------------------------------------------------//
/*!
 * Bethe-Heitler relativistic pair production from gammas.
 */
struct BetheHeitlerModel
{
};

//---------------------------------------------------------------------------//
/*!
 * Pair production from muons.
 *
 * \todo Move MuPPET table celeritas::ImportMuPairProductionTable here.
 */
struct MuPairProductionModel
{
};

//---------------------------------------------------------------------------//
// ALIASES
//---------------------------------------------------------------------------//
//!@{
//! \name Model aliases
//! \todo rename `em/model` to match

using MuBremsstrahlungModel = MuBremsModel;
using RelativisticBremModel = RelBremsModel;

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
