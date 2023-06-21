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
 * Isotopes of a given element are loaded sequentially and contiguously in
 * Geant4. Therefore, the \c IsotopeIdx pair is used to retrieve the available
 * isotopes of a given element, which are stored in \c ImportData::isotopes .
 * \c isotope_index.first represents the starting index in the
 * \c ImportData::isotopes vector, while \c isotope_index.second stores the
 * last index. Thus, to loop over the available isotope data:
 * \code
   ImportData data;
   // Load import data
   auto const& element = data.elements[3]; // E.g. select 3rd element
   auto const& key = element.isotope_index;
   for (auto i = key.first; i < key.second; i++)
   {
       auto frac = data.isotopes[i].fractional_abundance;
       // Do stuff
   }
 * \endcode
 */
struct ImportElement
{
    //!@{
    //! \name type aliases
    using VecIsotopeIdx = std::vector<int>;
    //!@}

    std::string name;
    int atomic_number;
    double atomic_mass;  //!< [atomic mass unit]
    double radiation_length_tsai;  //!< [g/cm^2]
    double coulomb_factor;
    VecIsotopeIdx isotope_indices;  //!< Indices in ImportData::isotopes
    std::vector<double> relative_abundance;  //!< Fractional abundance for this
                                             //!< element
};

//---------------------------------------------------------------------------//
/*!
 * Store isotope data.
 *
 * For nuclear mass, see `G4NucleiProperties::GetNuclearMass`.
 * For natural fractional abundance, see `G4NistManager::GetIsotopeAbundance`.
 */
struct ImportIsotope
{
    std::string name;  //!< Isotope label
    int atomic_number;  //!< Atomic number Z
    int atomic_mass_number;  //!< Atomic number A
    double nuclear_mass;  //!< [MeV]
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
