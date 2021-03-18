//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Constants.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Constants.hh"
#include "base/Units.hh"
#include "base/Quantity.hh"

namespace celeritas
{
namespace units
{
//---------------------------------------------------------------------------//
//! Special case for annotating that values are in the native unit system
struct NativeUnit
{
    //! The conversion factor of the resulting unit is always unity
    static CELER_CONSTEXPR_FUNCTION real_type value() { return 1; }
};

//! Unit for quantity such that the numeric value of 1 MeV is unity
struct Mev
{
    //! Conversion factor from the unit to CGS
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return 1e6 * constants::e_electron * units::volt;
    }
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
        return constants::e_electron; // *Positive* sign
    }
};

//! Unit for atomic masses
struct Amu
{
    //! Conversion factor from the unit to CGS
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return constants::atomic_mass;
    }
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
        return 1e-3 * units::barn;
    }
};

//! Unit for converting mass to an energy-valued quantity
using CLightSq = UnitProduct<CLight, CLight>;

//---------------------------------------------------------------------------//
//!@{
//! Units for particle quantities
using ElementaryCharge = Quantity<EElectron>;
using MevEnergy        = Quantity<Mev>;
using LogMevEnergy     = Quantity<LogMev>;
using MevMass          = Quantity<UnitDivide<Mev, CLightSq>>;
using MevMomentum      = Quantity<UnitDivide<Mev, CLight>>;
using MevMomentumSq    = Quantity<UnitDivide<UnitProduct<Mev, Mev>, CLightSq>>;
using LightSpeed       = Quantity<CLight>;
using AmuMass          = Quantity<Amu>;
//!@}

//---------------------------------------------------------------------------//
} // namespace units
} // namespace celeritas
