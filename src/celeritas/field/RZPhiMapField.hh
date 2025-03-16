//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZPhiMapField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Constants.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/grid/FindInterp.hh"
#include "corecel/grid/NonuniformGrid.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/math/Turn.hh"
#include "celeritas/Types.hh"

#include "RZPhiMapFieldData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate the value of magnetic field based on a volume-based RZPhi field
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

    NonuniformGrid<real_type> const grid_r_;
    NonuniformGrid<real_type> const grid_z_;
    NonuniformGrid<real_type> const grid_phi_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the shared magnetic field map data.
 */
CELER_FUNCTION
RZPhiMapField::RZPhiMapField(FieldParamsRef const& params)
    : params_{params}
    , grid_r_{params_.grids.r, params_.grids.storage}
    , grid_z_{params_.grids.z, params_.grids.storage}
    , grid_phi_{params_.grids.phi, params_.grids.storage}
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
    // Ensure phi is in [0, 2\f$\pi\f$)
    real_type phi = std::fmod(atan2(pos[1], pos[0]), 2 * constants::pi.value());
    auto turn_phi{native_value_to<Turn>(phi)};

    // Check if point is within field map bounds
    if (!params_.valid(pos[2], r, turn_phi))
        return value;

    // Find interpolation points for given r, z, and phi
    auto [ir, wr1] = find_interp<NonuniformGrid<real_type>>(grid_r_, r);
    auto [iz, wz1] = find_interp<NonuniformGrid<real_type>>(grid_z_, pos[2]);
    auto [iphi, wphi1]
        = find_interp<NonuniformGrid<real_type>>(grid_phi_, turn_phi.value());

    auto get_field = [this](size_type iz, size_type ir, size_type iphi) {
        return params_.fieldmap[params_.id(iphi, ir, iz)];
    };

    // Get the eight corner values for Z component of the field
    // clang-format off
    real_type v000 = get_field(iz,     ir,     iphi    ).value_z;
    real_type v001 = get_field(iz,     ir,     iphi + 1).value_z;
    real_type v010 = get_field(iz,     ir + 1, iphi    ).value_z;
    real_type v011 = get_field(iz,     ir + 1, iphi + 1).value_z;
    real_type v100 = get_field(iz + 1, ir,     iphi    ).value_z;
    real_type v101 = get_field(iz + 1, ir,     iphi + 1).value_z;
    real_type v110 = get_field(iz + 1, ir + 1, iphi    ).value_z;
    real_type v111 = get_field(iz + 1, ir + 1, iphi + 1).value_z;
    // clang-format on

    // Trilinear interpolation formula for Z component
    value[2] = (1 - wz1)
                   * ((1 - wr1) * ((1 - wphi1) * v000 + wphi1 * v001)
                      + wr1 * ((1 - wphi1) * v010 + wphi1 * v011))
               + wz1
                     * ((1 - wr1) * ((1 - wphi1) * v100 + wphi1 * v101)
                        + wr1 * ((1 - wphi1) * v110 + wphi1 * v111));

    // Get the eight corner values for R component of the field
    // clang-format off
    v000 = get_field(iz,     ir,     iphi    ).value_r;
    v001 = get_field(iz,     ir,     iphi + 1).value_r;
    v010 = get_field(iz,     ir + 1, iphi    ).value_r;
    v011 = get_field(iz,     ir + 1, iphi + 1).value_r;
    v100 = get_field(iz + 1, ir,     iphi    ).value_r;
    v101 = get_field(iz + 1, ir,     iphi + 1).value_r;
    v110 = get_field(iz + 1, ir + 1, iphi    ).value_r;
    v111 = get_field(iz + 1, ir + 1, iphi + 1).value_r;
    // clang-format on

    // Interpolate for R component
    real_type field_r = (1 - wz1)
                            * ((1 - wr1) * ((1 - wphi1) * v000 + wphi1 * v001)
                               + wr1 * ((1 - wphi1) * v010 + wphi1 * v011))
                        + wz1
                              * ((1 - wr1) * ((1 - wphi1) * v100 + wphi1 * v101)
                                 + wr1 * ((1 - wphi1) * v110 + wphi1 * v111));

    // Get the eight corner values for Phi component of the field
    // clang-format off
    v000 = get_field(iz,     ir,     iphi    ).value_phi;
    v001 = get_field(iz,     ir,     iphi + 1).value_phi;
    v010 = get_field(iz,     ir + 1, iphi    ).value_phi;
    v011 = get_field(iz,     ir + 1, iphi + 1).value_phi;
    v100 = get_field(iz + 1, ir,     iphi    ).value_phi;
    v101 = get_field(iz + 1, ir,     iphi + 1).value_phi;
    v110 = get_field(iz + 1, ir + 1, iphi    ).value_phi;
    v111 = get_field(iz + 1, ir + 1, iphi + 1).value_phi;
    // clang-format on

    // Interpolate for Phi component
    real_type field_phi
        = (1 - wz1)
              * ((1 - wr1) * ((1 - wphi1) * v000 + wphi1 * v001)
                 + wr1 * ((1 - wphi1) * v010 + wphi1 * v011))
          + wz1
                * ((1 - wr1) * ((1 - wphi1) * v100 + wphi1 * v101)
                   + wr1 * ((1 - wphi1) * v110 + wphi1 * v111));

    // Project cylindrical components to Cartesian coordinates
    // default for r == 0
    real_type cos_phi = 1.;
    real_type sin_phi = 0.;

    if (r != 0)
    {
        cos_phi = pos[0] / r;
        sin_phi = pos[1] / r;
    }
    value[0] = field_r * cos_phi - field_phi * sin_phi;
    value[1] = field_r * sin_phi + field_phi * cos_phi;

    return value;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas