//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportElement.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store element data.
 *
 * \c IsotopeIndex maps the isotope in the \c ImportData::isotopes vector.
 */
struct ImportElement
{
    //!@{
    //! \name type aliases
    using IsotopeIndex = int;
    using IsotopeFrac = std::pair<IsotopeIndex, double>;
    using VecIsotopeFrac = std::vector<IsotopeFrac>;
    //!@}

    std::string name;
    int atomic_number;
    double atomic_mass;  //!< [atomic mass unit]
    double radiation_length_tsai;  //!< [g/cm^2]
    double coulomb_factor;
    VecIsotopeFrac isotopes_fractions;  //!< Isotopic fractional abundance
};

//---------------------------------------------------------------------------//
/*!
 * Store isotope data.
 *
 * For nuclear mass, see `G4NucleiProperties::GetNuclearMass(int A, int Z)`.
 */
struct ImportIsotope
{
    std::string name;  //!< Isotope label
    int atomic_number;  //!< Atomic number Z
    int atomic_mass_number;  //!< Atomic number A
    double nuclear_mass;  //!< Sum of nucleons' mass + binding energy [MeV]
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
