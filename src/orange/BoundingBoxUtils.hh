//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBoxUtils.hh
//! \brief Utilities for bounding boxes
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <iosfwd>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/grid/GridTypes.hh"
#include "corecel/math/Algorithms.hh"
#include "geocel/BoundingBox.hh"

#include "OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class Translation;
class Transformation;

//---------------------------------------------------------------------------//
/*!
 * Check if a bounding box spans (-inf, inf) in every direction.
 */
template<class T>
inline bool is_infinite(BoundingBox<T> const& bbox)
{
    auto axes = range(Axis::size_);
    return all_of(axes.begin(), axes.end(), [&bbox](Axis ax) {
        constexpr T inf = numeric_limits<T>::infinity();
        return bbox.point(Bound::lo, ax) == -inf
               && bbox.point(Bound::hi, ax) == inf;
    });
}

//---------------------------------------------------------------------------//
/*!
 * Check if a bounding box has no infinities.
 *
 * \pre The bounding box cannot be null
 */
template<class T>
inline bool is_finite(BoundingBox<T> const& bbox)
{
    CELER_EXPECT(bbox);

    auto axes = range(Axis::size_);
    return all_of(axes.begin(), axes.end(), [&bbox](Axis ax) {
        return !std::isinf(bbox.point(Bound::lo, ax))
               && !std::isinf(bbox.point(Bound::hi, ax));
    });
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

    auto axes = range(Axis::size_);
    return any_of(axes.begin(), axes.end(), [&bbox](Axis ax) {
        return bbox.point(Bound::lo, ax) == bbox.point(Bound::hi, ax);
    });
}

//---------------------------------------------------------------------------//
/*!
 * Whether any axis has an infinity on one bound but not the other.
 */
template<class T>
inline bool is_half_inf(BoundingBox<T> const& bbox)
{
    auto axes = range(Axis::size_);
    return any_of(axes.begin(), axes.end(), [&bbox](Axis ax) {
        return std::isinf(bbox.point(Bound::lo, ax))
               != std::isinf(bbox.point(Bound::hi, ax));
    });
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the center of a bounding box.
 *
 * \pre The bounding box cannot be null, or "semi-infinite" (i.e., it may not
 * have a finite lower/upper value in a particular dimension, with a
 * corresponding infinite upper/lower value).
 */
template<class T>
inline Array<T, 3> calc_center(BoundingBox<T> const& bbox)
{
    CELER_EXPECT(bbox);
    CELER_EXPECT(!is_half_inf(bbox));

    Array<T, 3> center;
    for (auto ax : range(Axis::size_))
    {
        center[to_int(ax)]
            = (bbox.point(Bound::lo, ax) + bbox.point(Bound::hi, ax)) / 2;
        if (CELER_UNLIKELY(std::isnan(center[to_int(ax)])))
        {
            // Infinite or half-infinite
            center[to_int(ax)] = 0;
        }
    }

    return center;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the half widths of the bounding box.
 *
 * \pre The bounding box cannot be null
 */
template<class T>
inline Array<T, 3> calc_half_widths(BoundingBox<T> const& bbox)
{
    CELER_EXPECT(bbox);

    Array<T, 3> hw;
    for (auto ax : range(Axis::size_))
    {
        hw[to_int(ax)]
            = (bbox.point(Bound::hi, ax) - bbox.point(Bound::lo, ax)) / 2;
    }

    return hw;
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

    for (auto ax : range(Axis::size_))
    {
        lengths[to_int(ax)] = bbox.point(Bound::hi, ax)
                              - bbox.point(Bound::lo, ax);
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

    for (auto ax : range(Axis::size_))
    {
        result *= bbox.point(Bound::hi, ax) - bbox.point(Bound::lo, ax);
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

    for (auto ax : range(Axis::size_))
    {
        lower[to_int(ax)]
            = celeritas::min(a.point(Bound::lo, ax), b.point(Bound::lo, ax));
        upper[to_int(ax)]
            = celeritas::max(a.point(Bound::hi, ax), b.point(Bound::hi, ax));
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

    for (auto ax : range(Axis::size_))
    {
        lower[to_int(ax)]
            = celeritas::max(a.point(Bound::lo, ax), b.point(Bound::lo, ax));
        upper[to_int(ax)]
            = celeritas::min(a.point(Bound::hi, ax), b.point(Bound::hi, ax));
    }

    return BoundingBox<T>::from_unchecked(lower, upper);
}

//---------------------------------------------------------------------------//
/*!
 * Check if all points inside the small bbox are in the big bbox.
 *
 * All bounding boxes should enclose a "null" bounding box (there are no points
 * in the null box, so no points are outside the big box). The null bounding
 * box will enclose no real bounding boxes. Comparing two null bounding boxes
 * is unspecified (forbidden for now).
 */
template<class T>
inline bool encloses(BoundingBox<T> const& big, BoundingBox<T> const& small)
{
    CELER_EXPECT(big || small);

    auto axes = range(Axis::size_);
    return all_of(axes.begin(), axes.end(), [&big, &small](Axis ax) {
        return big.point(Bound::lo, ax) <= small.point(Bound::lo, ax)
               && big.point(Bound::hi, ax) >= small.point(Bound::hi, ax);
    });
}

//---------------------------------------------------------------------------//
/*!
 * Check if a segment from \c pos in direction \c dir of length \c distance
 * intersects the bounding box.
 *
 * If the position is already inside the bounding box, the result is always
 * true. This function employs the slab method
 * \citep{kay-slab-1986, https://doi.org/10.1145/15886.15916}.
 */
template<class T>
inline CELER_FUNCTION bool intersects_segment(BoundingBox<T> const& bbox,
                                              Array<T, 3> const& pos,
                                              Array<T, 3> const& dir,
                                              T distance)
{
    T max_entry = 0;
    T min_exit = distance;

    // Loop over all three slab pairs to calculate the maximum distance
    // required to enter the regions between each slab pair and the minimum
    // distance to leave these regions
    for (auto ax : range(Axis::size_))
    {
        // Calculate the inverse of the direction for this axis. Note that we
        // do not have to check for dir != 0; we can rely on IEEE arithmetic to
        // provide values of +/-inf for inv_dirs, leading to +/-inf slab
        // distances that provide the correct behavior.
        T inv_dir = 1 / dir[to_int(ax)];

        // Calculate the entry/exit distance for this slab pair
        T entry = (bbox.point(Bound::lo, ax) - pos[to_int(ax)]) * inv_dir;
        T exit = (bbox.point(Bound::hi, ax) - pos[to_int(ax)]) * inv_dir;
        if (entry > exit)
        {
            // Entry is actually exit; swap values
            trivial_swap(entry, exit);
        }

        max_entry = celeritas::max(max_entry, entry);
        min_exit = celeritas::min(min_exit, exit);
    }

    return max_entry <= min_exit;
}

//---------------------------------------------------------------------------//
/*!
 * Bump a bounding box outward and possibly convert to another type.
 * \tparam T destination type
 * \tparam U source type
 *
 * The upper and lower coordinates are bumped outward independently using the
 * relative and absolute tolerances. To ensure that the outward bump is
 * not truncated in the destination type, the "std::nextafter" function
 * advances to the next floating point representable number.
 */
template<class T, class U = T>
class BoundingBoxBumper
{
  public:
    //!@{
    //! \name Type aliases
    using TolU = Tolerance<U>;
    using result_type = BoundingBox<T>;
    using argument_type = BoundingBox<U>;
    //!@}

  public:
    //! Construct with default "soft equal" tolerances
    BoundingBoxBumper() : tol_{TolU::from_softequal()} {}

    //! Construct with ORANGE tolerances
    explicit BoundingBoxBumper(TolU const& tol) : tol_{tol}
    {
        CELER_EXPECT(tol_);
    }

    //! Return the expanded and converted bounding box
    result_type operator()(argument_type const& bbox)
    {
        Array<T, 3> lower;
        Array<T, 3> upper;

        for (auto ax : range(Axis::size_))
        {
            lower[to_int(ax)] = this->bumped<-1>(bbox.point(Bound::lo, ax));
            upper[to_int(ax)] = this->bumped<+1>(bbox.point(Bound::hi, ax));
        }

        return result_type::from_unchecked(lower, upper);
    }

  private:
    TolU tol_;

    //! Calculate the bump distance given a point: see detail::BumpCalculator
    template<int S>
    T bumped(U value) const
    {
        U bumped = value
                   + S * celeritas::max(tol_.abs, tol_.rel * std::fabs(value));
        return std::nextafter(static_cast<T>(bumped),
                              S * numeric_limits<T>::infinity());
    }
};

//---------------------------------------------------------------------------//
// Calculate the bounding box of a transformed box
BBox calc_transform(Translation const& tr, BBox const& a);

BBox calc_transform(Transformation const& tr, BBox const& a);

//---------------------------------------------------------------------------//
template<class T>
std::ostream& operator<<(std::ostream&, BoundingBox<T> const& bbox);

//---------------------------------------------------------------------------//
}  // namespace celeritas
