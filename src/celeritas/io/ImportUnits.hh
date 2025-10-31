//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * native unit system and the CLHEP/Geant4 values. Note that MeV quantities are
 * set to unity for this special case (i.e. they retain their energy scaling
 * and need to be wrapped with the \c MevEnergy quantity when used in
 * Celeritas).
 *
 * \todo Rename to ImportUnit
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
    mev_len_sq,  //!< [MeV-len^2]
    time,  //!< Time [time]
    inv_len_cb,  //!< Number density [1/len^3]
    len_time_sq_per_mass,  //!< Inverse pressure [len-time^2/mass]
    inv_mev,  //!< Inverse energy [1/MeV]
    size_,
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
