//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBoxUtils.cc
//---------------------------------------------------------------------------//
#include "BoundingBoxUtils.hh"

#include "corecel/Assert.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/NumericLimits.hh"

#include "transform/Transformation.hh"
#include "transform/Translation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the bounding box of a translated box.
 */
BBox calc_transform(Translation const& tr, BBox const& a)
{
    CELER_EXPECT(a);
    return {tr.transform_up(a.lower()), tr.transform_up(a.upper())};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the bounding box of a transformed box.
 *
 * This method has a custom implementation of GEMV for applying exact rotations
 * to infinities if the matrix representation simply should switches vector
 * entries or flips their sign. Without the complication, floating point
 * arithmetic results in NaN from multiplying zeroes (in the matrix) by the
 * infinite values.
 */
BBox calc_transform(Transformation const& tr, BBox const& a)
{
    CELER_EXPECT(a);
    using real_type = BBox::real_type;
    using Real3 = BBox::Real3;

    constexpr real_type inf = numeric_limits<real_type>::infinity();

    // Specialized GEMV that ignores zeros to correctly handle infinities
    auto rotate = [&r = tr.rotation()](Real3 const& x) {
        Real3 result;
        for (size_type i = 0; i != 3; ++i)
        {
            result[i] = 0;
            for (size_type j = 0; j != 3; ++j)
            {
                if (r[i][j] != real_type{0})
                {
                    result[i] += r[i][j] * x[j];
                }
            }
        }
        return result;
    };
    auto get_point = [&a](bool hi, Axis ax) {
        return (hi ? a.upper() : a.lower())[to_int(ax)];
    };

    Real3 lower{inf, inf, inf};
    Real3 upper{-inf, -inf, -inf};

    // Iterate through all bounding box corners and limit bbox based on
    // rotated extents
    for (auto hi_x : {false, true})
    {
        for (auto hi_y : {false, true})
        {
            for (auto hi_z : {false, true})
            {
                Real3 point = rotate({get_point(hi_x, Axis::x),
                                      get_point(hi_y, Axis::y),
                                      get_point(hi_z, Axis::z)});
                for (auto ax : range(to_int(Axis::size_)))
                {
                    lower[ax] = ::celeritas::min(lower[ax], point[ax]);
                    upper[ax] = ::celeritas::max(upper[ax], point[ax]);
                }
            }
        }
    }

    // Apply translation to rotated corners
    auto result = BBox::from_unchecked(lower + tr.translation(),
                                       upper + tr.translation());
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
