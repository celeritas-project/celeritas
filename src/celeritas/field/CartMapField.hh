//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/Types.hh"

#include "CartMapFieldData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interpolate a magnetic field vector on an x/y/z grid.
 */
class CartMapField
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = cartmap_real_type;
    using Real3 = Array<celeritas::real_type, 3>;
    using FieldParamsRef = NativeCRef<CartMapFieldParamsData>;
    using field_view_t = FieldParamsRef::view_t;
    //!@}

  public:
    // Construct with the shared map data
    inline CELER_FUNCTION explicit CartMapField(FieldParamsRef const& shared);

    // Evaluate the magnetic field value for the given position
    CELER_FUNCTION
    inline Real3 operator()(Real3 const& pos) const;

  private:
    field_view_t const& field_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the shared magnetic field map data.
 */
CELER_FUNCTION
CartMapField::CartMapField(FieldParamsRef const& shared)
    : field_{shared.get_view()}
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
CELER_FUNCTION auto CartMapField::operator()(Real3 const& pos) const -> Real3
{
    Real3 value;

    // delegate interpolation to Covfie
    auto bfield = field_.at(static_cast<real_type>(pos[0]),
                            static_cast<real_type>(pos[1]),
                            static_cast<real_type>(pos[2]));
    for (auto i :
         range(static_cast<std::underlying_type_t<CartAxis>>(CartAxis::size_)))
    {
        value[i] = bfield[i];
    }

    return value;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
