//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZPhiMapField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/grid/FindInterp.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

#include "RZPhiMapFieldData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate the value of magnetic field based on a volume-based RZ-Phi field
 * map.
 */
class RZPhiMapField
{
  public:
    //!@{
    //! \name Type aliases
    using Real3 = Array<real_type, 3>;
    using FieldParamsRef = NativeCRef<RZPhiMapFieldParamsData>;
    //!@}

  public:
    // Construct with the shared map data
    inline CELER_FUNCTION explicit RZPhiMapField(FieldParamsRef const& shared);

    // Evaluate the magnetic field value for the given position
    CELER_FUNCTION
    inline Real3 operator()(Real3 const& pos) const;

  private:
    // Shared constant field map
    FieldParamsRef const& params_;

    UniformGrid const grid_r_;
    UniformGrid const grid_z_;
    UniformGrid const grid_phi_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the shared magnetic field map data.
 */
CELER_FUNCTION
RZPhiMapField::RZPhiMapField(FieldParamsRef const& params)
    : params_(params)
    , grid_r_(params_.grids.data_r)
    , grid_z_(params_.grids.data_z)
    , grid_phi_(params_.grids.data_phi)
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
CELER_FUNCTION auto RZPhiMapField::operator()(Real3 const& pos) const -> Real3
{
    CELER_ENSURE(params_);

    Real3 value{0, 0, 0};

    // Convert Cartesian to cylindrical coordinates
    real_type r = hypot(pos[0], pos[1]);
    real_type phi = atan2(pos[1], pos[0]);

    // Ensure phi is in [0, 2π)
    if (phi < 0)
        phi += 2 * M_PI;

    if (!params_.valid(pos[2], r, phi))
        return value;

    // Find interpolation points for given r, z, and phi
    FindInterp<real_type> interp_r = find_interp<UniformGrid>(grid_r_, r);
    FindInterp<real_type> interp_z = find_interp<UniformGrid>(grid_z_, pos[2]);
    FindInterp<real_type> interp_phi = find_interp<UniformGrid>(grid_phi_, phi);

    size_type ir = interp_r.index;
    size_type iz = interp_z.index;
    size_type iphi = interp_phi.index;

    // Perform trilinear interpolation for each field component
    // Define the interpolation weights
    real_type wr0 = 1.0 - interp_r.fraction;
    real_type wr1 = interp_r.fraction;
    real_type wz0 = 1.0 - interp_z.fraction;
    real_type wz1 = interp_z.fraction;
    real_type wphi0 = 1.0 - interp_phi.fraction;
    real_type wphi1 = interp_phi.fraction;

    // Get the eight corner values for Z component of the field
    real_type v000 = params_.fieldmap[params_.id(iz, ir, iphi)].value_z;
    real_type v001 = params_.fieldmap[params_.id(iz, ir, iphi + 1)].value_z;
    real_type v010 = params_.fieldmap[params_.id(iz, ir + 1, iphi)].value_z;
    real_type v011 = params_.fieldmap[params_.id(iz, ir + 1, iphi + 1)].value_z;
    real_type v100 = params_.fieldmap[params_.id(iz + 1, ir, iphi)].value_z;
    real_type v101 = params_.fieldmap[params_.id(iz + 1, ir, iphi + 1)].value_z;
    real_type v110 = params_.fieldmap[params_.id(iz + 1, ir + 1, iphi)].value_z;
    real_type v111
        = params_.fieldmap[params_.id(iz + 1, ir + 1, iphi + 1)].value_z;

    // Trilinear interpolation formula for Z component
    value[2] = wz0
                   * (wr0 * (wphi0 * v000 + wphi1 * v001)
                      + wr1 * (wphi0 * v010 + wphi1 * v011))
               + wz1
                     * (wr0 * (wphi0 * v100 + wphi1 * v101)
                        + wr1 * (wphi0 * v110 + wphi1 * v111));

    // Get the eight corner values for R component of the field
    v000 = params_.fieldmap[params_.id(iz, ir, iphi)].value_r;
    v001 = params_.fieldmap[params_.id(iz, ir, iphi + 1)].value_r;
    v010 = params_.fieldmap[params_.id(iz, ir + 1, iphi)].value_r;
    v011 = params_.fieldmap[params_.id(iz, ir + 1, iphi + 1)].value_r;
    v100 = params_.fieldmap[params_.id(iz + 1, ir, iphi)].value_r;
    v101 = params_.fieldmap[params_.id(iz + 1, ir, iphi + 1)].value_r;
    v110 = params_.fieldmap[params_.id(iz + 1, ir + 1, iphi)].value_r;
    v111 = params_.fieldmap[params_.id(iz + 1, ir + 1, iphi + 1)].value_r;

    // Interpolate for R component
    real_type field_r = wz0
                            * (wr0 * (wphi0 * v000 + wphi1 * v001)
                               + wr1 * (wphi0 * v010 + wphi1 * v011))
                        + wz1
                              * (wr0 * (wphi0 * v100 + wphi1 * v101)
                                 + wr1 * (wphi0 * v110 + wphi1 * v111));

    // Get the eight corner values for Phi component of the field
    v000 = params_.fieldmap[params_.id(iz, ir, iphi)].value_phi;
    v001 = params_.fieldmap[params_.id(iz, ir, iphi + 1)].value_phi;
    v010 = params_.fieldmap[params_.id(iz, ir + 1, iphi)].value_phi;
    v011 = params_.fieldmap[params_.id(iz, ir + 1, iphi + 1)].value_phi;
    v100 = params_.fieldmap[params_.id(iz + 1, ir, iphi)].value_phi;
    v101 = params_.fieldmap[params_.id(iz + 1, ir, iphi + 1)].value_phi;
    v110 = params_.fieldmap[params_.id(iz + 1, ir + 1, iphi)].value_phi;
    v111 = params_.fieldmap[params_.id(iz + 1, ir + 1, iphi + 1)].value_phi;

    // Interpolate for Phi component
    real_type field_phi = wz0
                              * (wr0 * (wphi0 * v000 + wphi1 * v001)
                                 + wr1 * (wphi0 * v010 + wphi1 * v011))
                          + wz1
                                * (wr0 * (wphi0 * v100 + wphi1 * v101)
                                   + wr1 * (wphi0 * v110 + wphi1 * v111));

    // Project cylindrical components to Cartesian coordinates
    real_type cos_phi = cos(phi);
    real_type sin_phi = sin(phi);

    // If r is zero, we can't normalize the radial component
    if (r != 0)
    {
        value[0] = field_r * cos_phi - field_phi * sin_phi;
        value[1] = field_r * sin_phi + field_phi * cos_phi;
    }
    else
    {
        // At r=0, the phi direction is undefined, so we just use the first
        // value
        value[0] = field_r;
        value[1] = field_phi;
    }

    return value;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
