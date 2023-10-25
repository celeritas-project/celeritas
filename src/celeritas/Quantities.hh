//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Quantities.hh
//! \brief Unit struct definitions for use with the Quantity class
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Quantity.hh"  // IWYU pragma: export

#include "Constants.hh"
#include "Units.hh"

namespace celeritas
{
namespace units
{
//---------------------------------------------------------------------------//
//! Special case for annotating that values are in the native unit system
struct NativeUnit
{
    //! The conversion factor of the resulting unit is always unity
    static CELER_CONSTEXPR_FUNCTION int value() { return 1; }
};

//! Unit for quantity such that the numeric value of 1 MeV is unity
struct Mev
{
    //! Conversion factor from the unit to CGS
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return real_type(1e6) * constants::e_electron * units::volt;
    }
    //! Text label for output
    static char const* label() { return "MeV"; }
};

//! Unit for annotating the logarithm of (E/MeV)
struct LogMev
{
    //! Conversion factor is not multiplicative
    static CELER_CONSTEXPR_FUNCTION real_type value() { return 1; }
};

//! Unit for relativistic speeds
struct CLight
{
    //! Conversion factor from the unit to CGS
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return constants::c_light;
    }
};

//! Unit for precise representation of particle charges
struct EElectron
{
    //! Conversion factor from the unit to CGS
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return constants::e_electron;  // *Positive* sign
    }
    //! Text label for output
    static char const* label() { return "e"; }
};

//! Unit for atomic masses
struct Amu
{
    //! Conversion factor from the unit to CGS
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return constants::atomic_mass;
    }
    //! Text label for output
    static char const* label() { return "amu"; }
};

//! Unit for cross sections
struct Barn
{
    //! Conversion factor from the unit to CGS
    static CELER_CONSTEXPR_FUNCTION real_type value() { return units::barn; }
};

//! Unit for cross sections
struct Millibarn
{
    //! Conversion factor from the unit to CGS
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return real_type(1e-3) * units::barn;
    }
};

//! Unit for field
struct Tesla
{
    //! Conversion factor from the unit to CGS
    static CELER_CONSTEXPR_FUNCTION real_type value() { return units::tesla; }
};

//---------------------------------------------------------------------------//
//!@{
//! \name Derivative units

struct CLightSq : UnitProduct<CLight, CLight>
{
    static char const* label() { return "c^2"; }
};
struct MevPerCsq : UnitDivide<Mev, CLightSq>
{
    static char const* label() { return "MeV/c^2"; }
};
//!@}

//---------------------------------------------------------------------------//
//!@{
//! \name Units for particle quantities
using ElementaryCharge = Quantity<EElectron>;
using MevEnergy = Quantity<Mev>;
using LogMevEnergy = Quantity<LogMev>;
using MevMass = Quantity<MevPerCsq>;
using MevMomentum = Quantity<UnitDivide<Mev, CLight>>;
using MevMomentumSq = Quantity<UnitDivide<UnitProduct<Mev, Mev>, CLightSq>>;
using LightSpeed = Quantity<CLight>;
using AmuMass = Quantity<Amu>;
using FieldTesla = Quantity<Tesla>;
//!@}

//---------------------------------------------------------------------------//
}  // namespace units
}  // namespace celeritas
