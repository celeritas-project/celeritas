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
 */
struct ImportElement
{
    std::string name;
    int atomic_number;
    double atomic_mass;  //!< [atomic mass unit]
    double radiation_length_tsai;  //!< [g/cm^2]
    double coulomb_factor;
    std::vector<int> isotope_index;  //!< Index in ImportData::isotopes vector
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
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
