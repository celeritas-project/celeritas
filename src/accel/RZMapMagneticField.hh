//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RZMapMagneticField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/field/RZMapField.hh"
#include "celeritas/field/RZMapFieldParams.hh"
#include "celeritas/g4/MagneticField.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Geant4 magnetic field class
using RZMapMagneticField
    = celeritas::MagneticField<RZMapFieldParams, RZMapField>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
