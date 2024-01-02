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
struct MevPerCsq
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return Mev::value() / (constants::c_light * constants::c_light);
    }
    static char const* label() { return "MeV/c^2"; }
};

//! Magnitude of electron charge (positive sign)
struct EElectron
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
#if CELERITAS_UNITS == CELERITAS_UNITS_CLHEP
        return units::e_electron;  // Mathematically equal to constants
#else
        return constants::e_electron;
#endif
    }
    static char const* label() { return "e"; }
};

//! Unit mass in CLHEP system
struct ClhepUnitMass
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        // Quantities before 1e6 are unity in CLHEP system
        constexpr auto kg_clhep = (Mev::value() * (nanosecond * nanosecond)
                                   / (millimeter * millimeter)
                                   / EElectron::value())
                                  * (1e6 * coulomb);
        return kg_clhep / kilogram;
    }
    static char const* label() { return "m_clhep"; }
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
//! \name CGS and SI units

struct Centimeter
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return units::centimeter;
    }
    static char const* label() { return "cm"; }
};

struct Meter
{
    static CELER_CONSTEXPR_FUNCTION real_type value() { return units::meter; }
    static char const* label() { return "m"; }
};

struct Gram
{
    static CELER_CONSTEXPR_FUNCTION real_type value() { return units::gram; }
    static char const* label() { return "g"; }
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
};

//! SI unit traits
template<>
struct UnitTraits<CELERITAS_UNITS_SI>
{
    using Length = Meter;
    using Mass = Kilogram;
    using Time = Second;
};

//! CLHEP unit traits
template<>
struct UnitTraits<CELERITAS_UNITS_CLHEP>
{
    using Length = Millimeter;
    using Mass = ClhepUnitMass;
    using Time = Nanosecond;
};

using CgsTraits = UnitTraits<CELERITAS_UNITS_CGS>;
using SiTraits = UnitTraits<CELERITAS_UNITS_SI>;
using ClhepTraits = UnitTraits<CELERITAS_UNITS_CLHEP>;
using NativeTraits = UnitTraits<CELERITAS_UNITS>;

//---------------------------------------------------------------------------//
}  // namespace units
}  // namespace celeritas
