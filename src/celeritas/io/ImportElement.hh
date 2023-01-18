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

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store element data.
 *
 * \sa ImportData
 */
struct ImportElement
{
    std::string name;
    int atomic_number;
    double atomic_mass;  //!< [atomic mass unit]
    double radiation_length_tsai;  //!< [g/cm^2]
    double coulomb_factor;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
