//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportUnits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Quantity of measure enumeration for imported data.
 *
 * These enumeration values are used to scale values between the Celeritas
 * native unit system and the CLHEP/Geant4 values.
 *
 * \todo Rename to ImportUnit??
 */
enum class ImportUnits
{
    unitless,  //!< No dimension
    mev,  //!< Energy [MeV]
    mev_per_len,  //!< Energy loss [MeV/len]
    len,  //!< Range [len]
    len_inv,  //!< Macroscopic xs [1/len]
    len_mev_inv,  //!< Scaled (1/E) macroscopic xs [1/len-MeV]
    mev_sq_per_len,  //!< Scaled [E^2] macroscopic xs  [MeV^2/len]
    len_sq,  //!< Microscopic cross section [len^2]
    time,  //!< Time [time]
    inv_len_cb,  //!< Number density [1/len^3]
    size_,
    // Deprecated aliases
    none = unitless,  //!< Deprecated
    mev_per_cm = mev_per_len,  //!< Deprecated
    cm = len,  //!< Deprecated
    cm_inv = len_inv,  //!< Deprecated
    cm_mev_inv = len_mev_inv,  //!< Deprecated
    mev_2_per_cm = mev_sq_per_len,  //!< Deprecated
    cm_2 = len_sq,  //!< Deprecated
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get the string label for units
char const* to_cstring(ImportUnits q);

// Get the multiplier to turn this quantity to a native value
double native_value_from(UnitSystem sys, ImportUnits q);

// Get the multiplier to turn a unit Geant4 value to a native value
double native_value_from_clhep(ImportUnits q);

//---------------------------------------------------------------------------//
}  // namespace celeritas
