//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapField.covfie.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Types.hh"

#include "RZMapFieldData.covfie.hh"  // IWYU pragma: keep

#include "detail/CovfieRZFieldTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate the value of magnetic field based on a volume-based RZ field map.
 *
 * Values outside the grid are returned as zero.
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
    using field_view_t =
        typename detail::CovfieRZFieldTraits<MemSpace::native>::field_t::view_t;
    ParamsRef const& shared_;
    field_view_t const& field_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the shared magnetic field map data.
 */
CELER_FUNCTION
RZMapField::RZMapField(ParamsRef const& shared)
    : shared_{shared}, field_{get_rz_map_field_view(shared)}
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnetic field vector for the given position.
 *
 * This queries the covfie 2D field at (r, z) to obtain (Br, Bz), then
 * projects Br onto the x and y components using the position direction.
 * The result is in the native Celeritas unit system.
 */
CELER_FUNCTION auto RZMapField::operator()(Real3 const& pos) const -> Real3
{
    using traits_t = detail::CovfieRZFieldTraits<MemSpace::native>;
    celeritas::real_type r = hypot(pos[0], pos[1]);

    // Return zero outside the grid
    if (r < shared_.min_r || r > shared_.max_r || pos[2] < shared_.min_z
        || pos[2] > shared_.max_z)
    {
        return {0, 0, 0};
    }

    auto bvec = traits_t::to_array(
        field_.at(static_cast<real_type>(r), static_cast<real_type>(pos[2])));

    // bvec = {Br, Bz}
    Real3 value;
    value[2] = bvec[1];

    celeritas::real_type br_over_r = (r != 0) ? bvec[0] / r
                                              : celeritas::real_type(0);
    value[0] = br_over_r * pos[0];
    value[1] = br_over_r * pos[1];

    return value;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
