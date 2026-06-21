//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Field.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_set>
#include <variant>
#include <vector>

#include "corecel/math/Turn.hh"
#include "geocel/Types.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/RZMapFieldInput.hh"

class G4LogicalVolume;

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Build a problem without magnetic fields.
 */
struct NoField
{
};

//---------------------------------------------------------------------------//
/*!
 * Create a uniform nonzero field.
 *
 * If volumes are specified, the field will only be present in those volumes.
 *
 * \todo Field driver options will be separate from the magnetic field. They,
 * plus the field type, will be specified in a FieldParams that maps
 * region/particle/energy to field setup. NOTE ALSO that \c
 * driver_options.max_substeps is redundant with \c
 * p.tracking.limits.field_substeps .
 */
struct UniformField
{
    using SetVolume = std::unordered_set<G4LogicalVolume const*>;
    using SetString = std::unordered_set<std::string>;
    using VariantSetVolume = std::variant<std::monostate, SetVolume, SetString>;

    //! Default field units are tesla
    UnitSystem units{UnitSystem::si};

    //! Field strength
    Real3 strength{0, 0, 0};

    //! Field driver options
    FieldDriverOptions driver_options;

    //! Volumes where the field is present (optional)
    VariantSetVolume volumes;
};

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
 * \todo Driver options should be outside field
 */
struct CylMapField
{
    using Turn = Turn_t<double>;

    //! r, phi, and z grid points
    std::vector<double> grid_r;
    std::vector<Turn> grid_phi;
    std::vector<double> grid_z;

    //! Flattened R-Phi-Z field component [bfield]
    std::vector<double> field;

    //! \todo Remove from field input; should be a separate input
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
/*!
 * Grid specification for a single axis.
 *
 * This specifies the minimum and maximum coordinate values and the number of
 * grid points.
 */
template<class T>
struct AxisGrid
{
    T min{};
    T max{};
    size_type num{};

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
struct CartMapField
{
    //! Grid specification for each axis [len]
    AxisGrid<double> x;
    AxisGrid<double> y;
    AxisGrid<double> z;

    //! Flattened X-Y-Z field component [bfield]
    std::vector<double> field;

    //! \todo Remove from field input; should be a separate input
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
/*!
 * Build a separable R-Z magnetic field from a file.
 *
 * \todo v0.7 Move field input here
 */
using RZMapField = ::celeritas::RZMapFieldInput;

//---------------------------------------------------------------------------//
//! Field type
using Field
    = std::variant<NoField, UniformField, RZMapField, CylMapField, CartMapField>;

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
