//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZPhiMapFieldInput.hh
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
 * Input data for a magnetic R-Z-Phi vector field stored on an R-Z-Phi grid.
 *
 * The magnetic field is discretized at nodes on an R-Z-Phi grid, and at each
 * point the field vector is approximated by a 3-D vector in R-Z-Phi. The input
 * units of this field are in *NATIVE UNITS* (cm/gauss when CGS). An optional
 * \c _units field in the input can specify whether the input is in SI or CGS
 * units, with allowable values of "si", "cgs", or "clhep". The native CLHEP
 * unit strength is 1000*tesla.
 *
 * The field values are all indexed with phi having stride 1, R having stride
 * (num_grid_phi), and Z having stride (num_grid_r * num_grid_phi): [Z][R][Phi]
 */
struct RZPhiMapFieldInput
{
    unsigned int num_grid_z{};
    unsigned int num_grid_r{};
    unsigned int num_grid_phi{};
    double min_z{};  //!< Lower z coordinate [len]
    double max_z{};  //!< Last z coordinate [len]
    double min_r{};  //!< Lower r coordinate [len]
    double max_r{};  //!< Last r coordinate [len]
    double min_phi{};  //!< Lower phi coordinate [rad]
    double max_phi{};  //!< Last phi coordinate [rad]
    std::vector<double> field_z;  //!< Flattened Z field component [bfield]
    std::vector<double> field_r;  //!< Flattened R field component [bfield]
    std::vector<double> field_phi;  //!< Flattened Phi field component [bfield]

    // TODO: remove from field input; should be a separate input
    FieldDriverOptions driver_options;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        // clang-format off
        return (num_grid_z >= 2)
            && (num_grid_r >= 2)
            && (num_grid_phi >= 2)
            && (min_r >= 0)
            && (max_z > min_z)
            && (max_r > min_r)
            && (max_phi > min_phi)
            && (field_z.size() == num_grid_z * num_grid_r * num_grid_phi)
            && (field_r.size() == field_z.size())
            && (field_phi.size() == field_z.size());
        // clang-format on
    }
};

//---------------------------------------------------------------------------//
/*!
 * Helper to read the field from a file or stream.
 *
 * Example to read from a file:
 * \code
   RZPhiMapFieldInput inp;
   std::ifstream("foo.json") >> inp;
 * \endcode
 */
std::istream& operator>>(std::istream& is, RZPhiMapFieldInput&);

//---------------------------------------------------------------------------//
/*!
 * Helper to write the field to a file or stream.
 */
std::ostream& operator<<(std::ostream& os, RZPhiMapFieldInput const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
