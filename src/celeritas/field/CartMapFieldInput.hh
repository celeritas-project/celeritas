//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapFieldInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "celeritas/Types.hh"

#include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Input data for a magnetic X-Y-Z vector field stored on an X-Y-Z grid.
 *
 * The magnetic field is discretized at nodes on an X-Y-Z grid, and at each
 * point the field vector is approximated by a 3-D vector in X-Y-Z. The input
 * units of this field are in *NATIVE UNITS* (cm/gauss when CGS). An optional
 * \c _units field in the input can specify whether the input is in SI or CGS
 * units, with allowable values of "si", "cgs", or "clhep". The native CLHEP
 * unit strength is 1000*tesla.
 *
 * The field values are all indexed with Z having stride 1, Y having stride
 * (num_grid_z), and X having stride (num_grid_y * num_grid_z): [X][Y][Z]
 */
struct CartMapFieldInput
{
    real_type min_x;  //!< Minimum X grid point [len]
    real_type max_x;  //!< Maximum X grid point [len]
    size_type num_x;  //!< Number of X grid points

    real_type min_y;  //!< Minimum Y grid point [len]
    real_type max_y;  //!< Maximum Y grid point [len]
    size_type num_y;  //!< Number of Y grid points

    real_type min_z;  //!< Minimum Z grid point [len]
    real_type max_z;  //!< Maximum Z grid point [len]
    size_type num_z;  //!< Number of Z grid points

    std::vector<real_type> field;  //!< Flattened X-Y-Z field component
                                   //!< [bfield]

    // TODO: remove from field input; should be a separate input
    FieldDriverOptions driver_options;

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        // clang-format off
        return 
             (max_x >= min_x)
            && (num_x >= 2)
            && (max_y >= min_y)
            && (num_y >= 2)
            && (max_z >= min_z)
            && (num_z >= 2)
            && (field.size() == static_cast<size_type>(CartAxis::size_) * num_x * num_y * num_z);
        // clang-format on
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas