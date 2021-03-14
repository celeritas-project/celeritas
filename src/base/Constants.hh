//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Constants.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Types.hh"
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
 * System of Units, 9th ed., for definition of constants and how they relate to
 * the different units.
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
 * re_electron          | classic_electr_radius   | Classical electron radius
 * kcd_luminous         | [none]                  | Lumens per Watt
 * lambda_compton       | electron_Compton_length | Reduced Compton wavelength
 *
 * Some experimental physical constants are derived from the other physical
 * constants, but for consistency and clarity they are presented numerically
 * with the units provided in the CODATA 2018 dataset. The \c Constants.test.cc
 * unit tests compare the numerical value against the derivative values inside
 * the celeritas unit system. All experimental values include the final
 * (ususally two) imprecise digits; their precision is usually on the order of
 * \f$ 10^{-11} \f$.
 */

//!@{
//! Mathemetical constant
constexpr real_type pi = 3.14159265358979323846; // truncated
//!@}

//!@{
//! Physical constant with *exact* value as defined by SI
constexpr real_type c_light    = 299792458. * units::meter / units::second;
constexpr real_type h_planck   = 6.62607015e-34 * units::joule * units::second;
constexpr real_type e_electron = 1.602176634e-19 * units::coulomb;
constexpr real_type k_boltzmann  = 1.380649e-23 * units::joule / units::kelvin;
constexpr real_type na_avogadro  = 6.02214076e23;
constexpr real_type kcd_luminous = 683;
//!@}

//!@{
//! Exact derivative constant
constexpr real_type hbar_planck = h_planck / (2 * pi);
//!@}

//!@{
//! Experimental physical constant from CODATA 2018
constexpr real_type a0_bohr              = 5.29177210903e-11 * units::meter;
constexpr real_type alpha_fine_structure = 7.2973525693e-3;
constexpr real_type atomic_mass          = 1.66053906660e-24 * units::gram;
constexpr real_type electron_mass        = 9.1093837015e-28 * units::gram;
constexpr real_type eps_electric         = 8.8541878128e-12 * units::farad
                                   / units::meter;
constexpr real_type mu_magnetic = 1.25663706212e-6 * units::newton
                                  / (units::ampere * units::ampere);
constexpr real_type re_electron  = 2.8179403262e-15 * units::meter;
constexpr real_type rinf_rydberg = 10973731.568160 / units::meter;
constexpr real_type eh_hartree   = 4.3597447222071e-18 / units::meter;
constexpr real_type lambda_compton = 3.8615926796e-13 * units::meter;
//!@}

//---------------------------------------------------------------------------//
} // namespace constants
} // namespace celeritas
