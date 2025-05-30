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
};
//---------------------------------------------------------------------------//
/*!
 * Construct a physics process for electron/positron pair production.
 */
struct PairProductionProcess
{
    //! Bethe-Heitler pair production
    std::optional<BetheHeitlerModel> bethe_heitler;
    //! Muonic pair production
    std::optional<MuPairProductionModel> mu;
};

//---------------------------------------------------------------------------//
//!@{
//! \name Process aliases
//! \todo rename `em/model` to match, merge muon and electron proceses

using BremsstrahlungProcess = BremsProcess;
using GammaConversionProcess = PairProductionProcess;
using MuPairProductionProcess = PairProductionProcess;

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
