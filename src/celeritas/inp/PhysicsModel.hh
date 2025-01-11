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
struct SBBremsModel
{
};

struct RelBremsModel
{
    //! Account for LPM effect at very high energies
    bool enable_lpm{true};
};

struct MuBremsModel
{
};

//---------------------------------------------------------------------------//
//!@{
//! \name Model aliases
//! \todo rename `em/model` to match

using SBBremsModel = SeltzerBergerBremsModel;
using RelativisticBremModel = RelBremsModel;
using MuBremsstrahlungModel = MuBremsModel;

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
