//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RZMapMagneticField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/field/RZMapFieldParams.hh"
#include "celeritas/g4/MagneticField.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * On-the-fly field calculation with covfie using Celeritas data+units.
 *
 * This "adapter" implementation hides the covfie dependency from downstream
 * users.
 */
struct RZAdapterField
{
    HostCRef<RZMapFieldParamsData> const& data;

    Real3 operator()(Real3 const&) const;
};

//---------------------------------------------------------------------------//
//! Geant4 magnetic field class for R-Z cylindrically symmetric field
using RZMapMagneticField
    = celeritas::MagneticField<RZMapFieldParams, RZAdapterField>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
