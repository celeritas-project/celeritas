//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Quantities.hh
//! \brief Derivative unit classes and annotated Quantity values
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/math/Quantity.hh"  // IWYU pragma: export

#include "UnitTypes.hh"

namespace celeritas
{
namespace units
{
//---------------------------------------------------------------------------//
//!@{
//! \name Derivative units

//! "Natural units" for mass
struct MevPerCsq : UnitDivide<Mev, UnitProduct<CLight, CLight>>
{
    static char const* label() { return "MeV/c^2"; }
};

//! "Natural units" for momentum
struct MevPerC : UnitDivide<Mev, CLight>
{
    static char const* label() { return "MeV/c"; }
};

//! Tesla field strength
struct Tesla
{
    //! Conversion factor from the unit to native
    static CELER_CONSTEXPR_FUNCTION real_type value() { return units::tesla; }
    static char const* label() { return "T"; }
};

//! Inverse cubic centimeter for number densities
struct InvCentimeterCubed
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return 1 / (units::centimeter * units::centimeter * units::centimeter);
    }
    static char const* label() { return "1/cm^3"; }
};

//! Molar density
struct MolPerCentimeterCubed
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return constants::na_avogadro
               / (units::centimeter * units::centimeter * units::centimeter);
    }
    static char const* label() { return "mol/cm^3"; }
};

//!@}

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
using CmLength = Quantity<Centimeter>;
using InvCmXs = Quantity<UnitInverse<Centimeter>>;
using InvCcDensity = Quantity<InvCentimeterCubed>;
using MolCcDensity = Quantity<MolPerCentimeterCubed>;
using FieldTesla = Quantity<Tesla>;
//!@}

//---------------------------------------------------------------------------//
}  // namespace units
}  // namespace celeritas
