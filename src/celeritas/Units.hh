//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Units.hh
//! \brief Unit definitions
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "corecel/math/Constant.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Units in Celeritas for macro-scale quantities.
 *
 * Celeritas can be configured at build time to use different unit systems for
 * better compatibility with external libraries and applications. The
 * \c CELERITAS_UNITS CMake variable can be set to one of the following:
 * - `CELERITAS_UNITS_CGS` (default): use Gaussian CGS units
 * - `CELERITAS_UNITS_SI`: use SI units
 * - `CELERITAS_UNITS_CLHEP`: use the Geant4 high energy physics system (a mix
 *   of macro-scale and atomic-scale units)
 *
 * The following units have numerical values of 1 in the default Celeritas
 * system (Gaussian CGS) and are often seen in unit tests:
 * - cm for standard unit of length
 * - s for standard unit of time
 * - g for standard unit of mass
 * - G for standard unit of field strength
 *
 * Unless otherwise specified, the user-selected unit system is used for input
 * and
 * output numerical values. They are meant for macro-scale quantities coupling
 * the different code components of Celeritas.
 *
 * \note This system of units should be fully consistent so that constants can
 * be precisely defined. (E.g., you cannot define both MeV as 1 and Joule
 * as 1.) To express quantities in another system of units, such as MeV and
 * "natural" units, use the Quantity class.
 *
 * See also:
 *  - \c Constants.hh for constants defined in this unit system
 *  - \c physics/base/Units.hh for unit systems used by the physics
 *
 * Additionally:
 * - radians are used for measures of angle (unitless)
 * - steradians are used for measures of solid angle (unitless)
 */
namespace units
{
//---------------------------------------------------------------------------//

#define CELER_ICC inline constexpr Constant

#if CELERITAS_UNITS == CELERITAS_UNITS_CGS
//!@{
//! \name Units with numerical value defined to be 1 for CGS
CELER_ICC centimeter{1};  //!< Length
CELER_ICC gram{1};  //!< Mass
CELER_ICC second{1};  //!< Time
CELER_ICC gauss{1};  //!< Field strength
CELER_ICC kelvin{1};  //!< Temperature
//!@}

//!@{
//! \name Exact unit transformations to SI units
CELER_ICC meter = 100 * centimeter;
CELER_ICC kilogram = 1000 * gram;
CELER_ICC tesla = 10000 * gauss;
//!@}

//!@{
//! \name Exact unit transformations using SI unit definitions
CELER_ICC newton = kilogram * meter / (second * second);
CELER_ICC joule = newton * meter;
CELER_ICC coulomb = kilogram / (tesla * second);
CELER_ICC ampere = coulomb / second;
CELER_ICC volt = joule / coulomb;
CELER_ICC farad = coulomb / volt;
//!@}

//!@{
//! \name CLHEP units
CELER_ICC millimeter = Constant{0.1} * centimeter;
CELER_ICC nanosecond = Constant{1e-9} * second;
//!@}

#elif CELERITAS_UNITS == CELERITAS_UNITS_SI
//!@{
//! \name Units with numerical value defined to be 1 for SI
CELER_ICC second{1};  //!< Time
CELER_ICC meter{1};  //!< Length
CELER_ICC kilogram{1};  //!< Mass
CELER_ICC kelvin{1};  //!< Temperature
CELER_ICC coulomb{1};  //!< Charge
//!@}

//!@{
//! \name Exact unit transformations using SI unit definitions
CELER_ICC newton = kilogram * meter / (second * second);
CELER_ICC joule = newton * meter;
CELER_ICC volt = joule / coulomb;
CELER_ICC tesla = volt * second / (meter * meter);
CELER_ICC ampere = coulomb / second;
CELER_ICC farad = coulomb / volt;
//!@}

//!@{
//! \name CGS units
CELER_ICC gauss = Constant{1e-4} * tesla;
CELER_ICC centimeter = Constant{1e-2} * meter;
CELER_ICC gram = Constant{1e-3} * kilogram;
//!@}

//!@{
//! \name CLHEP units
CELER_ICC millimeter = Constant{1e-3} * meter;
CELER_ICC nanosecond = Constant{1e-9} * second;
//!@}

#elif CELERITAS_UNITS == CELERITAS_UNITS_CLHEP

//!@{
//! \name Units with numerical value defined to be 1 for CLHEP
CELER_ICC millimeter{1};  //!< Length
CELER_ICC megaelectronvolt{1};  //!< Energy
CELER_ICC nanosecond{1};  //!< Time
CELER_ICC e_electron{1};  //!< Charge
CELER_ICC kelvin{1};  //!< Temperature
//!@}

// Note: conversion constant is the value from SI 2019
CELER_ICC coulomb = e_electron / Constant{1.602176634e-19};
CELER_ICC volt = Constant{1e-6} * (megaelectronvolt / e_electron);
CELER_ICC joule = coulomb * volt;

CELER_ICC second = Constant{1e9} * nanosecond;
CELER_ICC meter = 1000 * millimeter;

CELER_ICC ampere = coulomb / second;
CELER_ICC farad = coulomb / volt;
CELER_ICC kilogram = joule * (second / meter) * (second / meter);
CELER_ICC tesla = volt * second / (meter * meter);
CELER_ICC newton = joule / meter;

//!@{
//! \name CGS-specific units
CELER_ICC centimeter = 10 * millimeter;
CELER_ICC gram = Constant{1e-3} * kilogram;
CELER_ICC gauss = Constant{1e-4} * tesla;
//!@}

#endif

//!@{
//! \name Other common units
CELER_ICC micrometer = Constant{1e-4} * centimeter;
CELER_ICC nanometer = Constant{1e-7} * centimeter;
CELER_ICC femtometer = Constant{1e-13} * centimeter;
CELER_ICC barn = Constant{1e-24} * centimeter * centimeter;
//!@}

#undef CELER_ICC

//---------------------------------------------------------------------------//
/*!
 * \name User-defined literals for units
 *
 * Usage:
 * \code
 * using namespace celeritas::units::literals;
 * auto length = 2.5_centimeter;
 * \endcode
 *
 * \note In headers, prefer explicit multiplication (e.g., \c 2.5 *
 * units::centimeter) or bring the namespace inside a function scope:
 * \code
 * using namespace celeritas::units::literals;
 * return 2.5_centimeter;
 * \endcode
 */
namespace literals
{

#define CELER_DEFINE_UNIT_UDL(SUFFIX, NAME)                            \
    CELER_CONSTEXPR_FUNCTION double operator""_##SUFFIX(long double v) \
    {                                                                  \
        return v * units::NAME;                                        \
    }                                                                  \
    CELER_CONSTEXPR_FUNCTION Constant operator""_##SUFFIX(             \
        unsigned long long int v)                                      \
    {                                                                  \
        return v * units::NAME;                                        \
    }                                                                  \
    CELER_CONSTEXPR_FUNCTION double operator""_##NAME(long double v)   \
    {                                                                  \
        return v * units::NAME;                                        \
    }                                                                  \
    CELER_CONSTEXPR_FUNCTION Constant operator""_##NAME(               \
        unsigned long long int v)                                      \
    {                                                                  \
        return v * units::NAME;                                        \
    }

CELER_DEFINE_UNIT_UDL(cm, centimeter)
CELER_DEFINE_UNIT_UDL(g, gram)
CELER_DEFINE_UNIT_UDL(s, second)
CELER_DEFINE_UNIT_UDL(G, gauss)
CELER_DEFINE_UNIT_UDL(K, kelvin)
CELER_DEFINE_UNIT_UDL(m, meter)
CELER_DEFINE_UNIT_UDL(kg, kilogram)
CELER_DEFINE_UNIT_UDL(T, tesla)
CELER_DEFINE_UNIT_UDL(N, newton)
CELER_DEFINE_UNIT_UDL(J, joule)
CELER_DEFINE_UNIT_UDL(C, coulomb)
CELER_DEFINE_UNIT_UDL(A, ampere)
CELER_DEFINE_UNIT_UDL(V, volt)
CELER_DEFINE_UNIT_UDL(F, farad)
CELER_DEFINE_UNIT_UDL(mm, millimeter)
CELER_DEFINE_UNIT_UDL(ns, nanosecond)
CELER_DEFINE_UNIT_UDL(um, micrometer)
CELER_DEFINE_UNIT_UDL(nm, nanometer)
CELER_DEFINE_UNIT_UDL(fm, femtometer)
CELER_DEFINE_UNIT_UDL(b, barn)

#undef CELER_DEFINE_UNIT_UDL
}  // namespace literals

//---------------------------------------------------------------------------//
}  // namespace units
}  // namespace celeritas
