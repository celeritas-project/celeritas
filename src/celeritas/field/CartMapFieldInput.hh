//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapFieldInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "geocel/Types.hh"
#include "celeritas/Types.hh"

#include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Grid specification for a single axis
template<class T>
struct AxisGrid
{
    T min{};  //!< Minimum coordinate value
    T max{};  //!< Maximum coordinate value
    size_type num{};  //!< Number of grid points

    //! Check if parameters are valid
    explicit operator bool() const { return max > min && num > 1; }
};

//---------------------------------------------------------------------------//
/*!
 * Input data for a magnetic X-Y-Z vector field stored on an X-Y-Z grid.
 *
 * The magnetic field is discretized at nodes on an X-Y-Z grid, and at each
 * point the field vector is approximated by a 3-D vector in X-Y-Z. The input
 * units of this field are in *NATIVE UNITS* (cm/gauss when CGS).
 *
 * The field values are all indexed with Z having stride 3, for the
 * 3-dimensional vector at that position, Y having stride (num_grid_z * 3), and
 * X having stride (num_grid_y * num_grid_z * 3): [X][Y][Z][3]
 */
struct CartMapFieldInput
{
    AxisGrid<real_type> x;  //!< X-axis grid specification [len]
    AxisGrid<real_type> y;  //!< Y-axis grid specification [len]
    AxisGrid<real_type> z;  //!< Z-axis grid specification [len]

    std::vector<real_type> field;  //!< Flattened X-Y-Z field component
                                   //!< [bfield]

    // TODO: remove from field input; should be a separate input
    FieldDriverOptions driver_options;

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return x && y && z
               && (field.size()
                   == static_cast<size_type>(Axis::size_) * x.num * y.num
                          * z.num);
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
