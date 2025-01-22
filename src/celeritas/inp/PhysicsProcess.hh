//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/PhysicsProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>

#include "PhysicsModel.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Construct a physics process for bremsstrahlung.
 */
struct BremsProcess
{
    std::optional<SeltzerBergerModel> sb{std::in_place};
    std::optional<RelBremsModel> rel{std::in_place};
    std::optional<MuBremsModel> mu;

    //! Use a unified relativistic/SB interactor
    bool combined_model{true};
    //! Use integral method for sampling discrete interaction length
    bool integral_xs{true};
};
//---------------------------------------------------------------------------//
/*!
 * Construct a physics process for electron/positron pair production.
 */
struct PairProdProcess
{
    //! Bethe-Heitler pair production
    std::optional<BetheHeitlerModel> bethe_heitler;
    //! Muonic pair production
    std::optional<MuPairProdModel> mu;
};

//---------------------------------------------------------------------------//
//!@{
//! \name Process aliases
//! \todo rename `em/model` to match, merge muon and electron proceses

using BremsstrahlungProcess = BremsProcess;
using GammaConversionProcess = PairProdProcess;
using MuPairProductionProcess = PairProdProcess;

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
