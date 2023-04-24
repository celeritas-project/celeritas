//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZFieldInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <vector>

#include "celeritas_config.h"

namespace celeritas
{
namespace test
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
struct RZFieldInput
{
    unsigned int num_grid_z{};
    unsigned int num_grid_r{};
    double delta_grid{};  //!< Grid spacing [cm]
    double offset_z{};  //!< Lower z coordinate [cm]
    std::vector<double> field_z;  //!< Flattened Z field component [tesla]
    std::vector<double> field_r;  //!< Flattened R field component [tesla]
};

//---------------------------------------------------------------------------//
/*!
 * Helper to read the field from a file or stream.
 *
 * Example to read from a file:
 * \code
   RZFieldInput inp;
   std::ifstream("foo.json") >> inp;
 * \endcode
 */
std::istream& operator>>(std::istream& is, RZFieldInput&);

//---------------------------------------------------------------------------//
/*!
 * Helper to write the field to a file or stream.
 */
std::ostream& operator<<(std::ostream& os, RZFieldInput const&);

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_JSON
inline std::istream& operator>>(std::istream&, RZFieldInput&)
{
    CELER_NOT_CONFIGURED("JSON");
}
inline std::ostream& operator<<(std::ostream&, RZFieldInput const&)
{
    CELER_NOT_CONFIGURED("JSON");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
