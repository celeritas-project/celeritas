//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBoxUtils.cc
//---------------------------------------------------------------------------//
#include "BoundingBoxUtils.hh"

#include <iostream>

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
 * Calculate the axis-aligned bounding box of a transformed box.
 *
 * Transforming a box usually results in an oriented bounding box, but
 * sometimes that OBB is also an AABB. This function implicitly creates the
 * transformed bounding box and creates an AABB that encompasses that box.
 *
 * The result returns an exactly transformed bounding box for axis-aligned
 * rotations. To achieve exactness for semi-infinite bounding boxes, this
 * method has a custom implementation of GEMV for applying exact rotations
 * if the matrix representation simply switches vector entries or flips their
 * sign. Without the complication, floating point arithmetic results in NaN
 * from multiplying zeroes (in the matrix) by the infinite values.
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
        for (auto i : range(3))
        {
            result[i] = 0;
            for (auto j : range(3))
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
                for (auto ax : range(3))
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
/*!
 * Write a bounding box to a stream.
 */
template<class T>
std::ostream& operator<<(std::ostream& os, BoundingBox<T> const& bbox)
{
    os << '{';
    if (bbox)
    {
        os << bbox.lower() << ", " << bbox.upper();
    }
    os << '}';
    return os;
}

template std::ostream& operator<<(std::ostream&, BoundingBox<float> const&);
template std::ostream& operator<<(std::ostream&, BoundingBox<double> const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
