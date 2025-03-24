//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CylMapFieldInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/math/SoftEqual.hh"
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
 * units of this field are in *NATIVE UNITS* (cm/gauss when CGS). An optional
 * \c _units field in the input can specify whether the input is in SI or CGS
 * units, with allowable values of "si", "cgs", or "clhep". The native CLHEP
 * unit strength is 1000*tesla.
 *
 * The field values are all indexed with Z having stride 1, Phi having stride
 * (num_grid_z), and R having stride (num_grid_phi * num_grid_z): [R][Phi][Z]
 */
struct CylMapFieldInput
{
    std::vector<real_type> grid_r;  //!< R grid points [len]
    std::vector<RealTurn> grid_phi;  //!< Phi grid points [AU]
    std::vector<real_type> grid_z;  //!< Z grid points [len]

    std::vector<real_type> field;  //!< Flattened R-Phi-Z field component
                                   //!< [bfield]

    // TODO: remove from field input; should be a separate input
    FieldDriverOptions driver_options;

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        // clang-format off
        return 
             (grid_r.size() >= 2)
            && (grid_phi.size() >= 2)
            && (grid_z.size() >= 2)
            && (grid_r.front() >= 0)
            && (soft_zero(grid_phi.front().value()))
            && (soft_equal(real_type{1}, grid_phi.back().value()))
            && std::is_sorted(grid_r.cbegin(), grid_r.cend())
            && std::is_sorted(grid_phi.cbegin(), grid_phi.cend())
            && std::is_sorted(grid_z.cbegin(), grid_z.cend())
            && (field.size() == static_cast<size_type>(CylAxis::size_) * grid_r.size() * grid_phi.size() * grid_z.size());
        // clang-format on
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas