//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/UnitTypes.hh
//! \brief Annotated unit struct definitions for use with Quantity
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/math/UnitUtils.hh"

#include "Constants.hh"
#include "Units.hh"

namespace celeritas
{
namespace units
{
//---------------------------------------------------------------------------//
//!@{
//! \name Natural units

struct CLight
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return constants::c_light;
    }
    static char const* label() { return "c"; }
};

//! "Natural units" for energy
struct Mev
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
#if CELERITAS_UNITS == CELERITAS_UNITS_CLHEP
        return units::megaelectronvolt;
#else
        return real_type(1e6) * constants::e_electron * units::volt;
#endif
    }
    static char const* label() { return "MeV"; }
};

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

//! Magnitude of electron charge (positive sign)
struct EElectron
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return constants::e_electron;
    }
    static char const* label() { return "e"; }
};

//! Quantity of substance
struct Mol
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return constants::na_avogadro;
    }
    static char const* label() { return "mol"; }
};

//!@}
//---------------------------------------------------------------------------//
//!@{
//! \name Atomic units

struct Amu
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return constants::atomic_mass;
    }
    static char const* label() { return "amu"; }
};

struct Barn
{
    static CELER_CONSTEXPR_FUNCTION real_type value() { return units::barn; }
    static char const* label() { return "b"; }
};

struct Millibarn
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return real_type(1e-3) * units::barn;
    }
    static char const* label() { return "mb"; }
};

//!@}
//---------------------------------------------------------------------------//
//!@{
//! \name Gaussian units

struct Centimeter
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return units::centimeter;
    }
    static char const* label() { return "cm"; }
};

struct Gram
{
    static CELER_CONSTEXPR_FUNCTION real_type value() { return units::gram; }
    static char const* label() { return "g"; }
};

struct Gauss
{
    static CELER_CONSTEXPR_FUNCTION real_type value() { return units::gauss; }
    static char const* label() { return "G"; }
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
struct MolPerCentimeterCubed : UnitProduct<Mol, InvCentimeterCubed>
{
    static char const* label() { return "mol/cm^3"; }
};

//!@}
//---------------------------------------------------------------------------//
//!@{
//! \name SI units

struct Meter
{
    static CELER_CONSTEXPR_FUNCTION real_type value() { return units::meter; }
    static char const* label() { return "m"; }
};

struct Kilogram
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return units::kilogram;
    }
    static char const* label() { return "kg"; }
};

struct Second
{
    static CELER_CONSTEXPR_FUNCTION real_type value() { return units::second; }
    static char const* label() { return "s"; }
};

struct Tesla
{
    static CELER_CONSTEXPR_FUNCTION real_type value() { return units::tesla; }
    static char const* label() { return "T"; }
};

//!@}
//---------------------------------------------------------------------------//
//!@{
//! \name CLHEP units

struct Millimeter
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return units::millimeter;
    }
    static char const* label() { return "mm"; }
};

struct Nanosecond
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return units::nanosecond;
    }
    static char const* label() { return "ns"; }
};

//! Unit mass in CLHEP system
struct ClhepUnitMass
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        if constexpr (CELERITAS_UNITS == CELERITAS_UNITS_CLHEP)
        {
            // Floating point errors make the true expression below difficult
            // to be exactly unity
            return real_type(1);
        }
        else
        {
            return constants::e_electron / coulomb * kilogram * real_type(1e-6);
        }
    }
    static char const* label() { return "mass_clhep"; }
};

//! Unit magnetic flux density in CLHEP system
struct ClhepUnitBField
{
    static CELER_CONSTEXPR_FUNCTION real_type value() { return 1e3 * tesla; }
    static char const* label() { return "field_clhep"; }
};

//!@}
//---------------------------------------------------------------------------//
//!@{
//! \name Annotation-only units

//! Mark as being in the native/builtin unit system
struct Native
{
    static CELER_CONSTEXPR_FUNCTION real_type value() { return 1; }
};

//! Annotate a quantity represented the logarithm of (E/MeV)
struct LogMev
{
    //! Conversion factor is not multiplicative
    static CELER_CONSTEXPR_FUNCTION real_type value() { return 0; }
};

//!@}
//---------------------------------------------------------------------------//

//! Traits class for units
template<int>
struct UnitTraits;

//! CGS unit traits
template<>
struct UnitTraits<CELERITAS_UNITS_CGS>
{
    using Length = Centimeter;
    using Mass = Gram;
    using Time = Second;
    using BField = Gauss;  //!< Magnetic flux density
};

//! SI unit traits
template<>
struct UnitTraits<CELERITAS_UNITS_SI>
{
    using Length = Meter;
    using Mass = Kilogram;
    using Time = Second;
    using BField = Tesla;
};

//! CLHEP unit traits
template<>
struct UnitTraits<CELERITAS_UNITS_CLHEP>
{
    using Length = Millimeter;
    using Mass = ClhepUnitMass;
    using Time = Nanosecond;
    using BField = ClhepUnitBField;
};

using CgsTraits = UnitTraits<CELERITAS_UNITS_CGS>;
using SiTraits = UnitTraits<CELERITAS_UNITS_SI>;
using ClhepTraits = UnitTraits<CELERITAS_UNITS_CLHEP>;
using NativeTraits = UnitTraits<CELERITAS_UNITS>;

//---------------------------------------------------------------------------//
}  // namespace units
}  // namespace celeritas
