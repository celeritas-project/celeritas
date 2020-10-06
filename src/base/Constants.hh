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
 */

//@{
//! Mathemetical constants
constexpr real_type pi = 3.14159265358979323846; // truncated
//@}

//@{
//! Physical constants with *exact* values as defined by SI
constexpr real_type c_light    = 299792458.0 * units::meter / units::second;
constexpr real_type h_planck   = 6.62607015e-34 * units::joule * units::second;
constexpr real_type e_electron = 1.602176634e-19 * units::coulomb;
constexpr real_type k_boltzmann  = 1.380649e-23 * units::joule / units::kelvin;
constexpr real_type na_avogadro  = 6.02214076e23; // [1/mol]
constexpr real_type kcd_luminous = 683;           // lm/W
//@}

//@{
//! Experimental physical constants
constexpr real_type alpha_fine_structure = 1 / 137.035999084; // CODATA 2018
constexpr real_type atomic_mass = 12. * units::gram / na_avogadro; // SI 2019
//@}

//@{
//! Derived physical constants
constexpr real_type eps_electric
    = e_electron * e_electron
      / (2 * alpha_fine_structure * h_planck * c_light); // vacuum permittivity
constexpr real_type mu_magnetic
    = 1 / (eps_electric * c_light * c_light); // vaccum permeability
//@}

//---------------------------------------------------------------------------//
} // namespace constants
} // namespace celeritas
