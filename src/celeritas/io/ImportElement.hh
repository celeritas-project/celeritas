//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
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
 * Store isotope data.
 *
 * For nuclear mass, see `G4NucleiProperties::GetNuclearMass(int A, int Z)`.
 */
struct ImportIsotope
{
    std::string name;  //!< Isotope label
    int atomic_number;  //!< Atomic number Z
    int atomic_mass_number;  //!< Atomic number A
    double binding_energy;  //!< Nuclear binding energy (BE) [MeV]
    double proton_loss_energy;  //!< BE(A, Z) - BE(A-1, Z-1) [MeV]
    double neutron_loss_energy;  //!< BE(A, Z) - BE(A-1, Z) [MeV]
    double nuclear_mass;  //!< Sum of nucleons' mass + binding energy [MeV]
};

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
    using IsotopeIndex = unsigned int;
    using IsotopeFrac = std::pair<IsotopeIndex, double>;
    using VecIsotopeFrac = std::vector<IsotopeFrac>;
    //!@}

    std::string name;
    int atomic_number;
    double atomic_mass;  //!< [amu]
    VecIsotopeFrac isotopes_fractions;  //!< Isotopic fractional abundance
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
