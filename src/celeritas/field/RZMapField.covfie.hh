//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapField.covfie.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Types.hh"

#include "RZMapFieldData.hh"  // IWYU pragma: keep

#include "detail/CovfieRZFieldTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate the value of magnetic field based on a volume-based RZ field map.
 *
 * Uses covfie for 2D bilinear interpolation on the R-Z grid and reconstructs
 * the 3D field vector by projecting the radial component onto Cartesian axes.
 */
class RZMapField
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = float;
    using Real3 = Array<celeritas::real_type, 3>;
    using ParamsRef = NativeCRef<RZMapFieldParamsData>;
    //!@}

  public:
    // Construct with the shared map data
    inline CELER_FUNCTION explicit RZMapField(ParamsRef const& shared);

    // Evaluate the magnetic field value for the given position
    CELER_FUNCTION
    inline Real3 operator()(Real3 const& pos) const;

  private:
    using field_view_t = ParamsRef::view_t;
    field_view_t const& field_;
    celeritas::real_type min_r_;
    celeritas::real_type max_r_;
    celeritas::real_type min_z_;
    celeritas::real_type max_z_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the shared magnetic field map data.
 */
CELER_FUNCTION
RZMapField::RZMapField(ParamsRef const& shared)
    : field_{shared.get_view()}
    , min_r_{shared.min_r}
    , max_r_{shared.max_r}
    , min_z_{shared.min_z}
    , max_z_{shared.max_z}
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnetic field vector for the given position.
 *
 * This uses covfie for 2-D bilinear interpolation on the input R-Z grid
 * and reconstructs the magnetic field vector from the stored R and Z
 * components of the field. Values outside the grid return zero, matching
 * the non-covfie implementation. The result is in the native Celeritas unit
 * system.
 */
CELER_FUNCTION auto RZMapField::operator()(Real3 const& pos) const -> Real3
{
    celeritas::real_type r = hypot(pos[0], pos[1]);

    // Return zero outside grid (matching non-covfie behavior)
    if (r < min_r_ || r > max_r_ || pos[2] < min_z_ || pos[2] > max_z_)
        return {0, 0, 0};

    // Covfie does 2D bilinear interpolation
    auto bfield = detail::CovfieRZFieldTraits<MemSpace::native>::to_real2(
        field_.at(static_cast<real_type>(r), static_cast<real_type>(pos[2])));
    celeritas::real_type br = bfield[0];
    celeritas::real_type bz = bfield[1];

    // Project Br onto Cartesian x/y components
    celeritas::real_type bx = (r != 0) ? br * pos[0] / r : 0;
    celeritas::real_type by = (r != 0) ? br * pos[1] / r : 0;

    return {bx, by, bz};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
