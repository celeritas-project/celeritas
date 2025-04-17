//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CovfieMapField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/math/Turn.hh"
#include "celeritas/Types.hh"

#include "CylMapFieldData.hh"

#include "detail/CovfieFieldType.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interpolate a magnetic field vector on an r/phi/z grid.
 *
 * The field vector is stored as a cartesian \f$(x,y,z)\f$ value on the
 * cylindrical mesh grid points, and trilinear interpolation is performed
 * within each grid cell. The value outside the grid is zero.
 *
 * Currently the grid requires a full \f$2\pi\f$ azimuthal grid.
 */
class CovfieMapField
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = cylmap_real_type;
    using Real3 = Array<celeritas::real_type, 3>;
    using CovfieField = CovfieFieldTrait<MemSpace::native>::field_t;
    //!@}

  public:
    // Construct with the shared map data
    inline CELER_FUNCTION explicit CovfieMapField(CovfieField const& field);

    // Evaluate the magnetic field value for the given position
    CELER_FUNCTION
    inline Real3 operator()(Real3 const& pos) const;

  private:
    CovfieField::view_t field_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the shared magnetic field map data.
 */
CELER_FUNCTION
CovfieMapField::CovfieMapField(CovfieField const& field) : field_{field} {}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnetic field vector for the given position.
 *
 * This does a 3-D interpolation on the input grid and reconstructs the
 * magnetic field vector from the stored R, Z, and Phi components of the field.
 * The result is in the native Celeritas unit system.
 */
CELER_FUNCTION auto CovfieMapField::operator()(Real3 const& pos) const -> Real3
{
    Array<real_type, 3> value{0, 0, 0};

    // Convert Cartesian to cylindrical coordinates
    real_type r = hypot(pos[0], pos[1]);
    // Ensure phi is in [0, 2\f$\pi\f$)
    auto phi = atan2turn<real_type>(pos[1], pos[0]);
    if (phi < zero_quantity())
    {
        phi.value() += 1;
    }
    if (CELER_UNLIKELY(phi.value() == 1))
    {
        // Make sure phi is in [0, 1). If phi is a negative value smaller
        // than machine epsilon, adding 1 will result in phi equal to 1
        phi.value() = 0;
    }
    CELER_ASSERT(phi >= zero_quantity() && phi.value() < 1);

    // delegate interpolation to Covfie
    auto bfield = field_.at(r, phi.value(), static_cast<real_type>(pos[2]));
    EnumArray<CylAxis, real_type> interp_field;
    for (auto i : range(CylAxis::size_))
    {
        interp_field[i]
            = bfield[static_cast<std::underlying_type_t<CylAxis>>(i)];
    }

    // Project cylindrical components to Cartesian coordinates
    // default for r == 0
    real_type cos_phi = 1;
    real_type sin_phi = 0;

    if (r != 0)
    {
        cos_phi = pos[0] / r;
        sin_phi = pos[1] / r;
    }
    value[0] = interp_field[CylAxis::r] * cos_phi
               - interp_field[CylAxis::phi] * sin_phi;
    value[1] = interp_field[CylAxis::r] * sin_phi
               + interp_field[CylAxis::phi] * cos_phi;
    value[2] = interp_field[CylAxis::z];

    return {value[0], value[1], value[2]};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
