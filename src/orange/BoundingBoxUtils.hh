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
    auto all_equal = [](Array<T, 3> const& values, T rhs) {
        return all_of(
            values.begin(), values.end(), [rhs](T lhs) { return lhs == rhs; });
    };
    constexpr T inf = numeric_limits<T>::infinity();

    return all_equal(bbox.lower(), -inf) && all_equal(bbox.upper(), inf);
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

    auto isinf = [](T value) { return std::isinf(value); };
    return !any_of(bbox.lower().begin(), bbox.lower().end(), isinf)
           && !any_of(bbox.upper().begin(), bbox.upper().end(), isinf);
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

    auto axes = range(to_int(Axis::size_));
    return any_of(axes.begin(), axes.end(), [&bbox](int ax) {
        return bbox.lower()[ax] == bbox.upper()[ax];
    });
}

//---------------------------------------------------------------------------//
/*!
 * Whether any axis has an infinity on one bound but not the other.
 */
template<class T>
inline bool is_half_inf(BoundingBox<T> const& bbox)
{
    auto axes = range(to_int(Axis::size_));
    return any_of(axes.begin(), axes.end(), [&bbox](int ax) {
        return std::isinf(bbox.lower()[ax]) != std::isinf(bbox.upper()[ax]);
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
    for (auto ax : range(to_int(Axis::size_)))
    {
        center[ax] = (bbox.lower()[ax] + bbox.upper()[ax]) / 2;
        if (CELER_UNLIKELY(std::isnan(center[ax])))
        {
            // Infinite or half-infinite
            center[ax] = 0;
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
    for (auto ax : range(to_int(Axis::size_)))
    {
        hw[ax] = (bbox.upper()[ax] - bbox.lower()[ax]) / 2;
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

    auto axes = range(to_int(Axis::size_));
    return all_of(axes.begin(), axes.end(), [&big, &small](int ax) {
        return big.lower()[ax] <= small.lower()[ax]
               && big.upper()[ax] >= small.upper()[ax];
    });
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the distance to the inside of the bbox from a pos and dir.
 *
 * The supplied position is expected to be outside of the bbox. If there is no
 * intersection, the result will be inf. This function employs the slab method
 * \citep{kay-slab-1986, https://doi.org/10.1145/15886.15916}.
 */
template<class T, class U>
inline CELER_FUNCTION T calc_dist_to_inside(BoundingBox<T> const& bbox,
                                            Array<U, 3> const& pos,
                                            Array<U, 3> const& dir)
{
    CELER_EXPECT(!is_inside(bbox, pos));

    T max_entry = 0;
    T min_exit = numeric_limits<T>::infinity();

    // Loop over all three slab pairs to calculate the maximum distance
    // required to enter the regions between each slab pair and the minimum
    // distance to leave these regions
    for (auto ax : range(to_int(Axis::size_)))
    {
        // Calculate the inverse of the direction for this axis. Note that we
        // do not have to check for dir != 0; we can rely on IEEE arithmetic to
        // provide values of +/-inf for inv_dirs, leading to +/-inf slab
        // distances that provide the correct behavior.
        T inv_dir = 1 / T(dir[ax]);

        // Calculate the entry/exit distance for this slab pair
        T entry = (bbox.lower()[ax] - T(pos[ax])) * inv_dir;
        T exit = (bbox.upper()[ax] - T(pos[ax])) * inv_dir;
        if (entry > exit)
        {
            // Entry is actually exit; swap values
            trivial_swap(entry, exit);
        }

        max_entry = celeritas::max(max_entry, entry);
        min_exit = celeritas::min(min_exit, exit);
    }

    // The distance to inside is the max entry, provided that it is greater
    // than the minimum exit distance
    return max_entry <= min_exit ? max_entry : numeric_limits<T>::infinity();
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

        for (auto ax : range(to_int(Axis::size_)))
        {
            lower[ax] = this->bumped<-1>(bbox.lower()[ax]);
            upper[ax] = this->bumped<+1>(bbox.upper()[ax]);
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
