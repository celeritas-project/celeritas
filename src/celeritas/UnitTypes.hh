//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/UnitTypes.hh
//! \brief Annotated unit struct definitions for use with Quantity
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "corecel/math/Constant.hh"
#include "corecel/math/UnitUtils.hh"

#include "Constants.hh"
#include "Types.hh"
#include "Units.hh"

namespace celeritas
{
namespace units
{
//! \cond (CELERITAS_DOC_DEV)
//---------------------------------------------------------------------------//
//!@{
//! \name Natural units

//! Natural unit of speed
struct CLight
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return constants::c_light;
    }
    static char const* label() { return "c"; }
};

//! Natural unit of charge (positive electron)
struct EElectron
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return constants::e_electron;
    }
    static char const* label() { return "e"; }
};

//!@}

//---------------------------------------------------------------------------//
//!@{
//! \name Atomic units

//! Atom-scale energy
struct ElectronVolt
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
#if CELERITAS_UNITS == CELERITAS_UNITS_CLHEP
        return units::megaelectronvolt / Constant(1e6);
#else
        return constants::e_electron * units::volt;
#endif
    }
    static char const* label() { return "eV"; }
};

//! Nucleus-scale energy
struct Mev
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
#if CELERITAS_UNITS == CELERITAS_UNITS_CLHEP
        return units::megaelectronvolt;
#else
        return Constant(1e6) * constants::e_electron * units::volt;
#endif
    }
    static char const* label() { return "MeV"; }
};

//! Nucleus-scale mass
struct MevPerCsq : UnitDivide<Mev, UnitProduct<CLight, CLight>>
{
    static char const* label() { return "MeV/c^2"; }
};

//! Nucleus-scale momentum
struct MevPerC : UnitDivide<Mev, CLight>
{
    static char const* label() { return "MeV/c"; }
};

//! Atomic mass units [amu]
struct Amu
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return constants::atomic_mass;
    }
    static char const* label() { return "amu"; }
};

//! Barn cross section [b]
struct Barn
{
    static CELER_CONSTEXPR_FUNCTION Constant value() { return units::barn; }
    static char const* label() { return "b"; }
};

//! Millibarn cross section [mb]
struct Millibarn
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return Constant(1e-3) * units::barn;
    }
    static char const* label() { return "mb"; }
};

//! Amount of substance \f$N_a\f$
struct Mol
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return constants::na_avogadro;
    }
    static char const* label() { return "mol"; }
};

//!@}

//---------------------------------------------------------------------------//
//!@{
//! \name Gaussian units for unit tests

struct Centimeter
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return units::centimeter;
    }
    static char const* label() { return "cm"; }
};

struct Gram
{
    static CELER_CONSTEXPR_FUNCTION Constant value() { return units::gram; }
    static char const* label() { return "g"; }
};

struct Gauss
{
    static CELER_CONSTEXPR_FUNCTION Constant value() { return units::gauss; }
    static char const* label() { return "G"; }
};

//! Inverse cubic centimeter for number densities
struct InvCentimeterCubed
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
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

//! Mass density
struct GramPerCentimeterCubed : UnitProduct<Gram, InvCentimeterCubed>
{
    static char const* label() { return "g/cm^3"; }
};

//!@}
//---------------------------------------------------------------------------//
//!@{
//! \name SI units

struct Meter
{
    static CELER_CONSTEXPR_FUNCTION Constant value() { return units::meter; }
    static char const* label() { return "m"; }
};

struct Kilogram
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return units::kilogram;
    }
    static char const* label() { return "kg"; }
};

struct Second
{
    static CELER_CONSTEXPR_FUNCTION Constant value() { return units::second; }
    static char const* label() { return "s"; }
};

struct Tesla
{
    static CELER_CONSTEXPR_FUNCTION Constant value() { return units::tesla; }
    static char const* label() { return "T"; }
};

//!@}

//---------------------------------------------------------------------------//
//!@{
//! \name CLHEP units

struct Millimeter
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return units::millimeter;
    }
    static char const* label() { return "mm"; }
};

struct Nanosecond
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return units::nanosecond;
    }
    static char const* label() { return "ns"; }
};

//! Unit mass in CLHEP system
struct ClhepUnitMass
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        if constexpr (CELERITAS_UNITS == CELERITAS_UNITS_CLHEP)
        {
            // Floating point errors make the true expression below difficult
            // to be exactly unity
            return Constant(1);
        }
        else
        {
            return constants::e_electron / coulomb * kilogram * Constant(1e-6);
        }
    }
    static char const* label() { return "mass_clhep"; }
};

//! Unit magnetic flux density in CLHEP system
struct ClhepUnitBField
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return Constant{1e3} * tesla;
    }
    static char const* label() { return "field_clhep"; }
};

//!@}

//---------------------------------------------------------------------------//
//!@{
//! \name Annotation-only units

//! Mark as being in the native/builtin unit system
struct Native
{
    static CELER_CONSTEXPR_FUNCTION Constant value() { return Constant{1}; }
};

//! Annotate a quantity represented the logarithm of (E/MeV)
struct LogMev
{
    //! Conversion factor is not multiplicative
    static CELER_CONSTEXPR_FUNCTION Constant value() { return Constant{0}; }
};

//!@}
//---------------------------------------------------------------------------//

//! Traits class for units
template<UnitSystem>
struct UnitSystemTraits;

//! \cond (BREATHE_FAILS_TO_PARSE)

//! CGS unit traits
template<>
struct UnitSystemTraits<UnitSystem::cgs>
{
    using Length = Centimeter;
    using Mass = Gram;
    using Time = Second;
    using BField = Gauss;  //!< Magnetic flux density

    static char const* label() { return "cgs"; }
};

//! SI unit traits
template<>
struct UnitSystemTraits<UnitSystem::si>
{
    using Length = Meter;
    using Mass = Kilogram;
    using Time = Second;
    using BField = Tesla;

    static char const* label() { return "si"; }
};

//! CLHEP unit traits
template<>
struct UnitSystemTraits<UnitSystem::clhep>
{
    using Length = Millimeter;
    using Mass = ClhepUnitMass;
    using Time = Nanosecond;
    using BField = ClhepUnitBField;

    static char const* label() { return "clhep"; }
};
//! \endcond

//!@{
//! \name Type aliases for unit system traits
using CgsTraits = UnitSystemTraits<UnitSystem::cgs>;
using SiTraits = UnitSystemTraits<UnitSystem::si>;
using ClhepTraits = UnitSystemTraits<UnitSystem::clhep>;
using NativeTraits = UnitSystemTraits<UnitSystem::native>;
//!@}

//---------------------------------------------------------------------------//
/*!
 * Expand a macro to a switch statement over all possible unit system types.
 *
 * This helper function is meant for processing user input to convert values to
 * the native unit system. It is *not* a \c CELER_FUNCTION because unit
 * conversion should be done only during preprocessing on the CPU.
 */
template<class F>
constexpr decltype(auto) visit_unit_system(F&& func, UnitSystem sys)
{
#define CELER_US_VISIT_CASE(TYPE) \
    case UnitSystem::TYPE:        \
        return std::forward<F>(func)(UnitSystemTraits<UnitSystem::TYPE>{});

    switch (sys)
    {
        CELER_US_VISIT_CASE(cgs);
        CELER_US_VISIT_CASE(si);
        CELER_US_VISIT_CASE(clhep);
        default:
            CELER_ASSERT_UNREACHABLE();
    }

#undef CELER_US_VISIT_CASE
}

//! \endcond
//---------------------------------------------------------------------------//
}  // namespace units
}  // namespace celeritas
