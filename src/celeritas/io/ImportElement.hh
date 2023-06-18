//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportElement.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store element data.
 *
 * The `isotope_index` pair is used to retrieve the list of available isotopes
 * of a given element: `isotope_index.first` represents the starting index in
 * the ImportData::isotopes vector, while `isotope_index.second` stores the
 * number of isotopes. Thus, to loop over the available isotope data:
 * \code
 * ImportData data;
 * // Load import data
 * auto const& element = data.elements[3]; // E.g. select 3rd element
 * auto const& key = element.isotope_index;
 * for (auto i = key.first; i < key.first + key.second; i++)
 * {
 *     auto frac = data.isotopes[i].fractional_abundance;
 *     // Do stuff
 * }
 * \endcode
 */
struct ImportElement
{
    //!@{
    //! \name type aliases
    using StartIdx = int;
    using NumIsotopes = int;
    using IsotopeIdx = std::pair<StartIdx, NumIsotopes>;
    //!@}
    std::string name;
    int atomic_number;
    double atomic_mass;  //!< [atomic mass unit]
    double radiation_length_tsai;  //!< [g/cm^2]
    double coulomb_factor;
    IsotopeIdx isotope_index;  //!< Pair of starting index and number of
                               //!< isotopes of this element in
                               //!< ImportData::isotopes vector
};

//---------------------------------------------------------------------------//
/*!
 * Store isotope data.
 */
struct ImportIsotope
{
    std::string name;  //!< Isotope label
    int atomic_number;  //!< Z number
    int atomic_mass_number;  //!< A number
    double nuclear_mass;  //!< TODO
    double fractional_abundance;  //!< Natural isotope abundance fraction
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
