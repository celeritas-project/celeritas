//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/grid/detail/FindInterp.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Units.hh"

#include "RZMapFieldData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate the value of magnetic field based on a volume-based RZ field map.
 */
class RZMapField
{
  public:
    //!@{
    //! \name Type aliases
    using Real3 = Array<real_type, 3>;
    using FieldParamsRef = NativeCRef<RZMapFieldParamsData>;
    //!@}

  public:
    // Construct with the shared map data
    CELER_FUNCTION
    explicit RZMapField(FieldParamsRef const& shared);

    // Evaluate the magnetic field value for the given position
    CELER_FUNCTION
    inline Real3 operator()(Real3 const& pos) const;

  private:
    // Shared constant field map
    FieldParamsRef const& params_;

    const UniformGrid grid_r_;
    const UniformGrid grid_z_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the shared magnetic field map data.
 */
CELER_FUNCTION
RZMapField::RZMapField(FieldParamsRef const& params)
    : params_(params)
    , grid_r_(params_.grids.data_r)
    , grid_z_(params_.grids.data_z)
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnetic field vector for the given position.
 *
 * This does a 2-D interpolation on the input grid and reconstructs the
 * magnetic field vector from the stored R and Z components of the field. The
 * result is in the native Celeritas unit system.
 */
CELER_FUNCTION auto RZMapField::operator()(Real3 const& pos) const -> Real3
{
    CELER_ENSURE(params_);

    Real3 value{0, 0, 0};

    real_type r = std::sqrt(ipow<2>(pos[0]) + ipow<2>(pos[1]));

    // Find interpolation points for given r and z
    detail::FindInterp<real_type> interp_r
        = detail::find_interp<UniformGrid>(grid_r_, r);
    detail::FindInterp<real_type> interp_z
        = detail::find_interp<UniformGrid>(grid_z_, pos[2]);

    size_type ir = interp_r.index;
    size_type iz = interp_z.index;

    if (!params_.valid(iz, ir))
        return value;

    // z component
    real_type low = params_.fieldmap[params_.id(iz, ir)].value_z;
    real_type high = params_.fieldmap[params_.id(iz + 1, ir)].value_z;
    value[2] = low + (high - low) * interp_z.fraction;

    // x and y components
    low = params_.fieldmap[params_.id(iz, ir)].value_r;
    high = params_.fieldmap[params_.id(iz, ir + 1)].value_r;
    real_type tmp = (r != 0) ? (low + (high - low) * interp_r.fraction) / r
                             : low;
    value[0] = tmp * pos[0];
    value[1] = tmp * pos[1];

    return value;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
