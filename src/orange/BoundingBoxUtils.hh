//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBoxUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/BoundingBox.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
class Translation;
class Transformation;
//---------------------------------------------------------------------------//
// Host/device functions
//---------------------------------------------------------------------------//
/*!
 * Determine if a point is contained in a bounding box.
 *
 * No point is ever contained in a null bounding box. A degenerate bounding
 * box will return "true" for any point on its face.
 */
template<class T, class U>
inline CELER_FUNCTION bool
is_inside(BoundingBox<T> const& bbox, Array<U, 3> point)
{
    constexpr auto axes = range(to_int(Axis::size_));
    return all_of(axes.begin(), axes.end(), [&point, &bbox](int ax) {
        return point[ax] >= bbox.lower()[ax] && point[ax] <= bbox.upper()[ax];
    });
}

//---------------------------------------------------------------------------//
// Host-only functions
//---------------------------------------------------------------------------//
/*!
 * Check if a bounding box spans (-inf, inf) in every direction.
 *
 * \pre The bounding box cannot be null
 */
template<class T>
inline bool is_infinite(BoundingBox<T> const& bbox)
{
    CELER_EXPECT(bbox);

    auto isinf = [](T value) { return std::isinf(value); };
    return all_of(bbox.lower().begin(), bbox.lower().end(), isinf)
           && all_of(bbox.upper().begin(), bbox.upper().end(), isinf);
}

//---------------------------------------------------------------------------//
/*!
 * Check if a bounding box has zero length in any direction.
 *
 * \pre The bounding box cannot be null
 */
template<class T>
inline bool is_degenerate(BoundingBox<T> const& bbox)
{
    CELER_EXPECT(bbox);

    constexpr auto axes = range(to_int(Axis::size_));
    return any_of(axes.begin(), axes.end(), [&bbox](int ax) {
        return bbox.lower()[ax] == bbox.upper()[ax];
    });
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the center of a bounding box.
 *
 * \pre The bounding box cannot be null
 */
template<class T>
inline Array<T, 3> calc_center(BoundingBox<T> const& bbox)
{
    CELER_EXPECT(bbox);

    Array<T, 3> center;
    for (auto ax : range(to_int(Axis::size_)))
    {
        center[ax] = (bbox.lower()[ax] + bbox.upper()[ax]) / 2;
    }

    return center;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the surface area of a bounding box.
 *
 * \pre The bounding box cannot be null
 */
template<class T>
inline T calc_surface_area(BoundingBox<T> const& bbox)
{
    CELER_EXPECT(bbox);

    Array<T, 3> lengths;

    for (auto ax : range(to_int(Axis::size_)))
    {
        lengths[ax] = bbox.upper()[ax] - bbox.lower()[ax];
    }

    return 2
           * (lengths[to_int(Axis::x)] * lengths[to_int(Axis::y)]
              + lengths[to_int(Axis::x)] * lengths[to_int(Axis::z)]
              + lengths[to_int(Axis::y)] * lengths[to_int(Axis::z)]);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the volume of a bounding box.
 *
 * \pre The bounding box cannot be null
 */
template<class T>
inline T calc_volume(BoundingBox<T> const& bbox)
{
    CELER_EXPECT(bbox);

    T result{1};

    for (auto ax : range(to_int(Axis::size_)))
    {
        result *= bbox.upper()[ax] - bbox.lower()[ax];
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the smallest bounding box enclosing two bounding boxes.
 */
template<class T>
inline constexpr BoundingBox<T>
calc_union(BoundingBox<T> const& a, BoundingBox<T> const& b)
{
    Array<T, 3> lower{};
    Array<T, 3> upper{};

    for (auto ax : range(to_int(Axis::size_)))
    {
        lower[ax] = celeritas::min(a.lower()[ax], b.lower()[ax]);
        upper[ax] = celeritas::max(a.upper()[ax], b.upper()[ax]);
    }

    return BoundingBox<T>::from_unchecked(lower, upper);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the intersection of two bounding boxes.
 *
 * If there is no intersection, the result will be a null bounding box.
 */
template<class T>
inline constexpr BoundingBox<T>
calc_intersection(BoundingBox<T> const& a, BoundingBox<T> const& b)
{
    Array<T, 3> lower{};
    Array<T, 3> upper{};

    for (auto ax : range(to_int(Axis::size_)))
    {
        lower[ax] = celeritas::max(a.lower()[ax], b.lower()[ax]);
        upper[ax] = celeritas::min(a.upper()[ax], b.upper()[ax]);
    }

    return BoundingBox<T>::from_unchecked(lower, upper);
}

//---------------------------------------------------------------------------//
/*!
 * Bump a bounding box outward and possibly convert to another type.
 * \tparam T destination type
 *
 * The upper and lower coordinates are bumped outward using the relative and
 * absolute tolerances
 */
template<class T>
class BoundingBoxBumper
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = BoundingBox<T>;
    //!@}

  public:
    //! Construct with default "soft equal" tolerances
    BoundingBoxBumper()
        : rel_{SoftEqual<T>{}.rel()}, abs_{SoftEqual<T>{}.abs()}
    {
    }

    //! Construct with a single bump tolerance used for both relative and abs
    explicit BoundingBoxBumper(T tol) : rel_{tol}, abs_{tol}
    {
        CELER_EXPECT(rel_ > 0 && abs_ > 0);
    }

    //! Construct with relative and absolute bump tolerances
    BoundingBoxBumper(T rel, T abs) : rel_{rel}, abs_{abs}
    {
        CELER_EXPECT(rel_ > 0 && abs_ > 0);
    }

    //! Return the expanded and converted bounding box
    template<class U>
    result_type operator()(BoundingBox<U> const& bbox)
    {
        CELER_EXPECT(bbox);

        Array<T, 3> lower;
        Array<T, 3> upper;

        for (auto ax : range(to_int(Axis::size_)))
        {
            T const lo = static_cast<T>(bbox.lower()[ax]);
            T const hi = static_cast<T>(bbox.upper()[ax]);
            T const bump = celeritas::max(abs_, rel_ * (hi - lo));
            lower[ax] = lo - bump;
            upper[ax] = hi + bump;
        }

        return result_type::from_unchecked(lower, upper);
    }

  private:
    T rel_;
    T abs_;
};

// Template deduction
template<class T>
BoundingBoxBumper(T) -> BoundingBoxBumper<T>;
template<class T>
BoundingBoxBumper(T, T) -> BoundingBoxBumper<T>;

//---------------------------------------------------------------------------//
// Calculate the bounding box of a transformed box
BBox calc_transform(Translation const& tr, BBox const& a);

BBox calc_transform(Transformation const& tr, BBox const& a);

//---------------------------------------------------------------------------//
}  // namespace celeritas
