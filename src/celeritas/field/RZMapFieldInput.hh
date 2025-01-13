//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <vector>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

#include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Input data for an magnetic R-Z vector field stored on an R-Z grid.
 *
 * The magnetic field is discretized at nodes on an R-Z grid, and each point
 * the field vector is approximated by a 2-D vector in R-Z. The input units of
 * this field are in *NATIVE UNITS* (cm/gauss when CGS). An optional \c _units
 * field in the input can specify whether the input is in SI or CGS
 * units, with allowable values of "si", "cgs", or "clhep". The native CLHEP
 * unit strength is 1000*tesla.
 *
 * The field values are all indexed with R having stride 1: [Z][R]
 *
 * \todo Use C indexing instead of Fortran? Or rename to ZR field?
 */
struct RZMapFieldInput
{
    unsigned int num_grid_z{};
    unsigned int num_grid_r{};
    double min_z{};  //!< Lower z coordinate [len]
    double max_z{};  //!< Last z coordinate [len]
    double min_r{};  //!< Lower r coordinate [len]
    double max_r{};  //!< Last r coordinate [len]
    std::vector<double> field_z;  //!< Flattened Z field component [bfield]
    std::vector<double> field_r;  //!< Flattened R field component [bfield]

    // TODO: remove from field input; should be a separate input
    FieldDriverOptions driver_options;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        // clang-format off
        return (num_grid_z >= 2)
            && (num_grid_r >= 2)
            && (min_r >= 0)
            && (max_z > min_z)
            && (max_r > min_r)
            && (field_z.size() == num_grid_z * num_grid_r)
            && (field_r.size() == field_z.size());
        // clang-format on
    }
};

//---------------------------------------------------------------------------//
/*!
 * Helper to read the field from a file or stream.
 *
 * Example to read from a file:
 * \code
   RZMapFieldInput inp;
   std::ifstream("foo.json") >> inp;
 * \endcode
 */
std::istream& operator>>(std::istream& is, RZMapFieldInput&);

//---------------------------------------------------------------------------//
/*!
 * Helper to write the field to a file or stream.
 */
std::ostream& operator<<(std::ostream& os, RZMapFieldInput const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
