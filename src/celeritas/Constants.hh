//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Constants.hh
//! \brief Mathematical, numerical, and physical constants
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

#include "Units.hh"

namespace celeritas
{
namespace constants
{
//---------------------------------------------------------------------------//
/*!
 * \namespace constants
 *
 * Mathematical, numerical, and physical constants. Some of the physical
 * constants listed here are *exact* numerical values: see the International
 * System of Units, 9th ed. (2019), for definition of constants and how they
 * relate to the different units.
 *
 * Celeritas            | CLHEP                   | Notes
 * -------------------- | ---------------------   | ------------
 * a0_bohr              | Bohr_radius             | Bohr radius
 * alpha_fine_structure | fine_structure_const    | |
 * atomic_mass          | amu                     | Not the same as 1/avogadro
 * eps_electric         | epsilon0                | Vacuum permittivity
 * h_planck             | h_Planck                | |
 * k_boltzmann          | k_Boltzmann             | |
 * mu_magnetic          | mu0                     | Vacuum permeability
 * na_avogadro          | Avogadro                | [1/mol]
 * r_electron           | classic_electr_radius   | Classical electron radius
 * kcd_luminous         | [none]                  | Lumens per Watt
 * lambdabar_electron   | electron_Compton_length | Reduced Compton wavelength
 *
 * Some experimental physical constants are derived from the other physical
 * constants, but for consistency and clarity they are presented numerically
 * with the units provided in the CODATA 2018 dataset. The \c Constants.test.cc
 * unit tests compare the numerical value against the derivative values inside
 * the celeritas unit system. All experimental values include the final
 * (ususally two) imprecise digits; their precision is usually on the order of
 * \f$ 10^{-11} \f$.
 */

#define CELER_ICRT inline constexpr real_type

//!@{
//! \name Mathemetical constants (truncated)
CELER_ICRT pi = 3.14159265358979323846;
CELER_ICRT euler = 2.71828182845904523536;
CELER_ICRT sqrt_two = 1.41421356237309504880;
CELER_ICRT sqrt_three = 1.73205080756887729353;
//!@}

//!@{
//! \name Physical constants with *exact* value as defined by SI
CELER_ICRT c_light = 299792458. * units::meter / units::second;
CELER_ICRT h_planck = 6.62607015e-34 * units::joule * units::second;
CELER_ICRT e_electron = 1.602176634e-19 * units::coulomb;
CELER_ICRT k_boltzmann = 1.380649e-23 * units::joule / units::kelvin;
CELER_ICRT na_avogadro = 6.02214076e23;
CELER_ICRT kcd_luminous = 683;
//!@}

//!@{
//! \name Exact derivative constants
CELER_ICRT hbar_planck = h_planck / (2 * pi);
//!@}

//!@{
//! \name Experimental physical constants from CODATA 2018
CELER_ICRT a0_bohr = 5.29177210903e-11 * units::meter;
CELER_ICRT alpha_fine_structure = 7.2973525693e-3;
CELER_ICRT atomic_mass = 1.66053906660e-24 * units::gram;
CELER_ICRT electron_mass = 9.1093837015e-28 * units::gram;
CELER_ICRT eps_electric = 8.8541878128e-12 * units::farad / units::meter;
CELER_ICRT mu_magnetic = 1.25663706212e-6 * units::newton
                         / (units::ampere * units::ampere);
CELER_ICRT r_electron = 2.8179403262e-15 * units::meter;
CELER_ICRT rinf_rydberg = 10973731.568160 / units::meter;
CELER_ICRT eh_hartree = 4.3597447222071e-18 / units::meter;
CELER_ICRT lambdabar_electron = 3.8615926796e-13 * units::meter;
//!@}

#undef CELER_ICRT

//---------------------------------------------------------------------------//
}  // namespace constants
}  // namespace celeritas
