//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Quantities.hh
//! \brief Derivative unit classes and annotated Quantity values
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/math/Quantity.hh"  // IWYU pragma: export

#include "UnitTypes.hh"  // IWYU pragma: export

namespace celeritas
{
namespace units
{
//---------------------------------------------------------------------------//
//!@{
//! \name Units for particle quantities
using ElementaryCharge = Quantity<EElectron>;
using MevEnergy = Quantity<Mev>;
using LogMevEnergy = Quantity<LogMev>;
using MevMass = Quantity<MevPerCsq>;
using MevMomentum = Quantity<MevPerC>;
using MevMomentumSq = Quantity<UnitProduct<MevPerC, MevPerC>>;
using LightSpeed = Quantity<CLight>;
using AmuMass = Quantity<Amu>;
//!@}

//---------------------------------------------------------------------------//
//!@{
//! \name Units for manual input and/or test harnesses
using BarnXs = Quantity<Barn>;
using CmLength = Quantity<Centimeter>;
using InvCmXs = Quantity<UnitInverse<Centimeter>>;
using InvCcDensity = Quantity<InvCentimeterCubed>;
using MolCcDensity = Quantity<MolPerCentimeterCubed>;
using GramCcDensity = Quantity<GramPerCentimeterCubed>;
using FieldTesla = Quantity<Tesla>;
//!@}
//---------------------------------------------------------------------------//
}  // namespace units
}  // namespace celeritas
