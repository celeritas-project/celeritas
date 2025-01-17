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
struct SBBremsModel
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
struct BHPairProdModel
{
};

//---------------------------------------------------------------------------//
/*!
 * Pair production from muons.
 *
 * \todo Move MuPPET table celeritas::ImportMuPairProductionTable here.
 */
struct MuPairProdModel
{
};

//---------------------------------------------------------------------------//
// ALIASES
//---------------------------------------------------------------------------//
//!@{
//! \name Model aliases
//! \todo rename `em/model` to match

#if 0
using BetheBlochModel       = BBIoniModel;
using BraggModel            = BraggIoniModel;
using CombinedBremModel     = SBRelBremModel;
using CoulombScatteringModel= WentzelScatModel;
using EPlusGGModel          = EPlusGGModel;
using ICRU73QOModel         = ICRU73QOModel;
using KleinNishinaModel     = KleinNishinaModel;
using LivermorePEModel      = LivermorePEModel;
using MollerBhabhaModel     = MollerBhabhaModel;
using MuBetheBlochModel     = MuBetheBlochModel;
using RayleighModel         = RayleighModel;
#endif

using BetheHeitlerModel = BHPairProdModel;
using MuBremsstrahlungModel = MuBremsModel;
using MuPairProductionModel = MuPairProdModel;
using RelativisticBremModel = RelBremsModel;
using SeltzerBergerModel = SBBremsModel;

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
