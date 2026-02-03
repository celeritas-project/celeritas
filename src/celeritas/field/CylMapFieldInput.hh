//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CylMapFieldInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/math/Turn.hh"
#include "celeritas/Types.hh"

#include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Input data for a magnetic R-Phi-Z vector field stored on an R-Phi-Z grid.
 *
 * The magnetic field is discretized at nodes on an R-Phi-Z grid, and at each
 * point the field vector is approximated by a 3-D vector in R-Phi-Z. The input
 * units of this field are in *NATIVE UNITS* (cm/gauss when CGS).
 *
 * The field values are all indexed with Z having stride 1, Phi having stride
 * (num_grid_z), and R having stride (num_grid_phi * num_grid_z): [R][Phi][Z]
 *
 * \todo Input should use double precision; driver options should be outside
 * field
 */
struct CylMapFieldInput
{
    std::vector<real_type> grid_r;  //!< R grid points [len]
    std::vector<RealTurn> grid_phi;  //!< Phi grid points [quantity]
    std::vector<real_type> grid_z;  //!< Z grid points [len]

    std::vector<real_type> field;  //!< Flattened R-Phi-Z field component
                                   //!< [bfield]

    // TODO: remove from field input; should be a separate input
    FieldDriverOptions driver_options;

    //! Whether grids have been assigned
    explicit operator bool() const
    {
        return grid_r.size() >= 2 && grid_phi.size() >= 2 && grid_z.size() >= 2
               && (field.size()
                   == static_cast<size_type>(CylAxis::size_) * grid_r.size()
                          * grid_phi.size() * grid_z.size());
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
