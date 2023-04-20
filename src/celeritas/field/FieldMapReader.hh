//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldMapReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "FieldMapData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load a RZ magnetic field map.
 *
 * A volume-based RZ magnetic field map stores r-z components of magnetic
 * field values of a cylindrical detector at the cm-unit grid point in the
 * range of r[0:FieldMapParameters::num_grid_r] and
 * z[-FieldMapParameters::num_grid_z:FieldMapParameters::num_grid_z]. The
 * format of input is FieldMapRecord where idx_r and idx_z are indices of
 * the 2-dimensional array[idx_z][idx_r], respectively.
 *
 */
class FieldMapReader
{
    //!@{
    //! \name Type aliases
    using result_type = FieldMapInput;
    //!@}

  public:
    // Construct the reader using the path of the map file
    FieldMapReader(FieldMapParameters const& params, std::string file_name);

    // Read volume-based RZ magnetic field map data
    result_type operator()() const;

  private:
    // Shared parameters for a user defined magnetic field map
    FieldMapParameters const& params_;
    // File name containing the magnetic field map
    std::string file_name_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
