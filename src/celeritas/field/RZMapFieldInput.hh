//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <vector>

#include "celeritas_config.h"
#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Input data for an magnetic R-Z vector field stored on an R-Z grid.
 *
 * The magnetic field is discretized at nodes on an R-Z grid, and each point
 * the field vector is approximated by a 2-D vector in R-Z. The input units of
 * this filed are in tesla.
 *
 * The field values are all indexed with R having stride 1: [Z][R]
 */
struct RZMapFieldInput
{
    unsigned int num_grid_z{};
    unsigned int num_grid_r{};
    double delta_grid{};  //!< Grid spacing [cm]
    double offset_z{};  //!< Offset of the lower z coordinate [cm]
    std::vector<double> field_z;  //!< Flattened Z field component [tesla]
    std::vector<double> field_r;  //!< Flattened R field component [tesla]

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        // clang-format off
        return (num_grid_z >= 2)
            && (num_grid_r >= 2)
            && (delta_grid > 0)
            && (offset_z >= 0)
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
#if !CELERITAS_USE_JSON
inline std::istream& operator>>(std::istream&, RZMapFieldInput&)
{
    CELER_NOT_CONFIGURED("JSON");
}
inline std::ostream& operator<<(std::ostream&, RZMapFieldInput const&)
{
    CELER_NOT_CONFIGURED("JSON");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
