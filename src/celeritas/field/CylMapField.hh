//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CylMapField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/grid/FindInterp.hh"
#include "corecel/grid/NonuniformGrid.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/math/Turn.hh"
#include "celeritas/Types.hh"

#include "CylMapFieldData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interpolate a magnetic field vector on an r/phi/z grid.
 *
 * The field vector is stored as cylindrical \f$(r,\phi,z)\f$ components on the
 * cylindrical mesh grid points, and trilinear interpolation is performed
 * within each grid cell. The value outside the grid is zero.
 *
 * Currently the grid requires a full \f$2\pi\f$ azimuthal grid.
 */
class CylMapField
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = cylmap_real_type;
    using Real3 = Array<celeritas::real_type, 3>;
    using ParamsRef = NativeCRef<CylMapFieldParamsData>;
    //!@}

  public:
    // Construct with the shared map data
    inline CELER_FUNCTION explicit CylMapField(ParamsRef const& shared);

    // Evaluate the magnetic field value for the given position
    CELER_FUNCTION
    inline Real3 operator()(Real3 const& pos) const;

  private:
    // Shared constant field map
    ParamsRef const& params_;

    NonuniformGrid<real_type> const grid_r_;
    NonuniformGrid<real_type> const grid_phi_;
    NonuniformGrid<real_type> const grid_z_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the shared magnetic field map data.
 */
CELER_FUNCTION
CylMapField::CylMapField(ParamsRef const& params)
    : params_{params}
    , grid_r_{params_.grids.axes[CylAxis::r], params_.grids.storage}
    , grid_phi_{params_.grids.axes[CylAxis::phi], params_.grids.storage}
    , grid_z_{params_.grids.axes[CylAxis::z], params_.grids.storage}
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnetic field vector for the given position.
 *
 * This does a 3-D interpolation on the input grid and reconstructs the
 * magnetic field vector from the stored R, Z, and Phi components of the field.
 * The result is in the native Celeritas unit system.
 */
CELER_FUNCTION auto CylMapField::operator()(Real3 const& pos) const -> Real3
{
    CELER_ENSURE(params_);

    Array<real_type, 3> value{0, 0, 0};

    // Convert Cartesian to cylindrical coordinates
    real_type r = hypot(pos[0], pos[1]);
    // Wrap turn from [-1/2; 1/2] into [0, 1)
    auto phi = atan2turn<real_type>(pos[1], pos[0]);
    phi.value() -= floor(phi.value());
    CELER_ASSERT(phi >= zero_quantity() && phi.value() < 1);

    // Check if point is within field map bounds
    if (!params_.valid(r, phi, pos[2]))
        return {0, 0, 0};

    // Find interpolation points for given r, phi, z
    auto [ir, wr1] = find_interp<NonuniformGrid<real_type>>(grid_r_, r);
    auto [iphi, wphi1]
        = find_interp<NonuniformGrid<real_type>>(grid_phi_, phi.value());
    auto [iz, wz1] = find_interp<NonuniformGrid<real_type>>(grid_z_, pos[2]);

    auto get_field = [this](size_type ir, size_type iphi, size_type iz) {
        return params_.fieldmap[params_.id(ir, iphi, iz)];
    };

    EnumArray<CylAxis, real_type> interp_field;

    for (auto axis : range(CylAxis::size_))
    {
        // clang-format off
        real_type v000 = get_field(ir,     iphi    ,     iz)[axis];
        real_type v001 = get_field(ir,     iphi    , iz + 1)[axis];
        real_type v010 = get_field(ir,     iphi + 1,     iz)[axis];
        real_type v011 = get_field(ir,     iphi + 1, iz + 1)[axis];
        real_type v100 = get_field(ir + 1, iphi    ,     iz)[axis];
        real_type v101 = get_field(ir + 1, iphi    , iz + 1)[axis];
        real_type v110 = get_field(ir + 1, iphi + 1,     iz)[axis];
        real_type v111 = get_field(ir + 1, iphi + 1, iz + 1)[axis];
        // clang-format on
        // Trilinear interpolation formula for the current component
        interp_field[axis]
            = (1 - wr1)
                  * ((1 - wphi1) * ((1 - wz1) * v000 + wz1 * v001)
                     + wphi1 * ((1 - wz1) * v010 + wz1 * v011))
              + wr1
                    * ((1 - wphi1) * ((1 - wz1) * v100 + wz1 * v101)
                       + wphi1 * ((1 - wz1) * v110 + wz1 * v111));
    }

    // Project cylindrical components to Cartesian coordinates
    real_type sin_phi;
    real_type cos_phi;
    sincos(phi, &sin_phi, &cos_phi);
    value[0] = interp_field[CylAxis::r] * cos_phi
               - interp_field[CylAxis::phi] * sin_phi;
    value[1] = interp_field[CylAxis::r] * sin_phi
               + interp_field[CylAxis::phi] * cos_phi;
    value[2] = interp_field[CylAxis::z];

    return {value[0], value[1], value[2]};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
