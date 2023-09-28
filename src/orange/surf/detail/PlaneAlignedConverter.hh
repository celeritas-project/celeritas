//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/PlaneAlignedConverter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>

#include "corecel/math/SoftEqual.hh"
#include "orange/surf/Plane.hh"
#include "orange/surf/PlaneAligned.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Try converting a plane to an aligned plane.
 */
class PlaneAlignedConverter
{
  public:
    // Construct with tolerance
    inline PlaneAlignedConverter(real_type tol);

    // Try converting to a plane with this orientation
    template<Axis T>
    std::optional<PlaneAligned<T>> operator()(AxisTag<T>, Plane const& p) const;

  private:
    SoftEqual<> soft_equal_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with tolerance.
 */
PlaneAlignedConverter::PlaneAlignedConverter(real_type tol) : soft_equal_{tol}
{
}

//---------------------------------------------------------------------------//
/*!
 * Try converting to an aligned plane.
 */
template<Axis T>
std::optional<PlaneAligned<T>>
PlaneAlignedConverter::operator()(AxisTag<T>, Plane const& p) const
{
    real_type const n = p.normal()[to_int(T)];
    if (!soft_equal_(real_type{1}, n))
    {
        // Not axis-aligned
        return {};
    }

    // Return the aligned plane, with displacment updated to give the same
    // position on the T axis
    return PlaneAligned<T>{p.displacement() / n};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
