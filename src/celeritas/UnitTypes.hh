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

struct EElectron
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return constants::e_electron;  // *Positive* sign
    }
    static char const* label() { return "e"; }
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

struct Gram
{
    static CELER_CONSTEXPR_FUNCTION real_type value() { return units::second; }
    static char const* label() { return "s"; }
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

//! CLHEP unit traits
template<>
struct UnitTraits<CELERITAS_UNITS_CLHEP>
{
    using Length = Millimeter;
    using Mass = Mev;
    using Time = Nanosecond;
};

using CgsTraits = UnitTraits<CELERITAS_UNITS_CGS>;
using ClhepTraits = UnitTraits<CELERITAS_UNITS_CLHEP>;
using NativeTraits = UnitTraits<CELERITAS_UNITS>;

//---------------------------------------------------------------------------//
}  // namespace units
}  // namespace celeritas
