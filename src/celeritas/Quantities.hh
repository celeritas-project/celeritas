//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Quantities.hh
//! \brief Derivative unit classes and annotated Quantity values
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Quantity.hh"  // IWYU pragma: export

#include "UnitTypes.hh"  // IWYU pragma: export

namespace celeritas
{
namespace units
{
//---------------------------------------------------------------------------//
//!@{
//! \name Quantities for atomic scale/natural units
using ElementaryCharge = RealQuantity<EElectron>;
using MevEnergy = RealQuantity<Mev>;
using MevMass = RealQuantity<MevPerCsq>;
using MevMomentum = RealQuantity<MevPerC>;
using MevMomentumSq = RealQuantity<UnitProduct<MevPerC, MevPerC>>;
using LightSpeed = RealQuantity<CLight>;
using AmuMass = RealQuantity<Amu>;
//!@}

//! Special faux quantity for overloading cross section calculation
using LogMevEnergy = RealQuantity<LogMev>;

//---------------------------------------------------------------------------//
//!@{
//! \name Quantities for manual input and/or test harnesses
using BarnXs = RealQuantity<Barn>;
using CmLength = RealQuantity<Centimeter>;
using InvCmXs = RealQuantity<UnitInverse<Centimeter>>;
using InvCcDensity = RealQuantity<InvCentimeterCubed>;
using MolCcDensity = RealQuantity<MolPerCentimeterCubed>;
using GramCcDensity = RealQuantity<GramPerCentimeterCubed>;
using FieldTesla = RealQuantity<Tesla>;
//!@}
//---------------------------------------------------------------------------//
}  // namespace units
}  // namespace celeritas
