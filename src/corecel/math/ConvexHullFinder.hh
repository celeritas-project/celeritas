//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/ConvexHullFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/SoftEqual.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find the convex hull of a sequence of 2D points.
 *
 * These points must be supplied in clockwise-order such that segments between
 * adjacent points, including the last and first points, comprise a
 * non-self-intersecting polygon. Exploiting this ordering, the Gram Scan
 * algorithm finds the convex hull with O(N) time complexity.
 */
template<class T>
class ConvexHullFinder
{
  public:
    //!@{
    //! \name Type aliases
    using Point = celeritas::Array<T, 2>;
    using Points = std::vector<Point>;
    using ConcaveRegions = std::vector<Points>;
    //!@}

  public:
    // Construct with vector of ordered points
    explicit ConvexHullFinder(Points const& points);

    // Return the convex hull
    Points convex_hull() const;

    // Calculate the concave regions, each supplied in clockwise order
    ConcaveRegions calc_concave_regions() const;

  private:
    /// TYPES ///
    using ConvexMask = std::vector<bool>;

    /// DATA ///
    Points const& points_;
    ConvexMask convex_mask_;
    size_type start_index_;
    SoftZero<T> soft_zero_;

    /// HELPER FUNCTIONS ///

    // Calculate a mask that indicates which points are on the convex hull
    ConvexMask calc_convex_mask() const;

    // Find the index of the element with the minimum y value
    size_type min_element_idx() const;

    // Determine if three points, traversed in order, form a clockwise turn
    bool is_clockwise(size_type i_prev, size_type i, size_type i_next) const;

    // Determine the next index, with modular indexing
    size_type calc_next(size_type i) const;

    // Determine the previous index, with modular indexing
    size_type calc_previous(size_type i) const;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with vector of ordered points.
 *
 * This function generates a mask that is used to calculate the convex hull
 * and associated concave regions. Note that this function does not enforce
 * ordering.
 *
 * \todo Check that points form a non-self-intersecting polygon with clockwise
 * ordering.
 */
template<class T>
ConvexHullFinder<T>::ConvexHullFinder(ConvexHullFinder::Points const& points)
    : points_{points}
{
    CELER_EXPECT(points_.size() > 2);
    start_index_ = this->min_element_idx();
    convex_mask_ = this->calc_convex_mask();
}

//---------------------------------------------------------------------------//
/*!
 * Return the convex hull.
 */
template<class T>
auto ConvexHullFinder<T>::convex_hull() const -> Points
{
    CELER_EXPECT(convex_mask_.size() > 2);

    Points convex_hull;
    for (auto i : range(convex_mask_.size()))
    {
        if (convex_mask_[i])
        {
            convex_hull.push_back(points_[i]);
        }
    }
    return convex_hull;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the concave regions, each supplied in clockwise order.
 *
 * Here, a "concave region" is a region that lies entirely within the convex
 * hull, that is concavity within the *original* shape. Note that a concave
 * region itself may be convex or concave. For example, consider the shape:
 *
 *   0 _______ 1
 *    |       |
 *    |     2 |____ 3
 *    |            |
 *  5 |____________| 4
 *
 * The convex hull is (0, 1, 3, 4, 5). There is one concave region: the
 * triangle formed by (1, 2, 3).
 */
template<class T>
auto ConvexHullFinder<T>::calc_concave_regions() const -> ConcaveRegions
{
    CELER_EXPECT(convex_mask_.size() > 2);
    ConcaveRegions concave_regions;

    // Since the original shape was supplied in clockwise order, we must
    // traverse the points backwards in order to obtain the concave regions in
    // clockwise order.
    size_type i = this->calc_previous(start_index_);
    while (i != start_index_)
    {
        if (convex_mask_[i])
        {
            i = this->calc_previous(i);
        }
        else
        {
            Points concave_region;
            concave_region.push_back(points_[calc_next(i)]);
            do
            {
                concave_region.push_back(points_[i]);
                i = this->calc_previous(i);
            } while (!convex_mask_[i]);

            concave_region.push_back(points_[i]);
            concave_regions.push_back(concave_region);
        }
    }
    return concave_regions;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Calculate a mask that indicates which points are on the convex hull.
 *
 * This method uses the Gram Scan algorithm.
 */
template<class T>
auto ConvexHullFinder<T>::calc_convex_mask() const -> ConvexMask
{
    // Find the indices of the points on the convex hull. Start from the point
    // with the lowest y value, which is gaurenteed to be on the hull.
    std::vector<size_type> hull;
    auto i = start_index_;
    hull.push_back(i);
    i = this->calc_next(i);

    for ([[maybe_unused]] auto _ : range(points_.size() - 1))
    {
        size_type i_next = this->calc_next(i);

        if (this->is_clockwise(hull.back(), i, i_next))
        {
            // Clockwise point is part of the hull
            hull.push_back(i);
        }
        else
        {
            // Pop points off the hull until we can reach the next point by
            // turning clockwise.
            while (hull.size() >= 2
                   && !this->is_clockwise(
                       hull[hull.size() - 2], hull.back(), i_next))
            {
                hull.pop_back();
            }
        }

        i = i_next;
    }

    // Convert convex hull indices to a mask
    ConvexMask convex_mask(points_.size(), false);
    for (auto h : hull)
    {
        convex_mask[h] = true;
    }

    return convex_mask;
}

//---------------------------------------------------------------------------//
/*!
 * Find the index of the element with the lowest y value.
 */
template<class T>
size_type ConvexHullFinder<T>::min_element_idx() const
{
    auto starting_it = std::min_element(
        points_.begin(), points_.end(), [](Point const& a, Point const& b) {
            return a[1] < b[1];
        });
    return std::distance(points_.begin(), starting_it);
};

//---------------------------------------------------------------------------//
/*!
 * Determine if three elements form a clockwise turn using the cross product.
 *
 * Here, colinear points are considered clockwise.
 */
template<class T>
auto ConvexHullFinder<T>::is_clockwise(size_type i_prev,
                                       size_type i,
                                       size_type i_next) const -> bool
{
    auto const& a = points_[i_prev];
    auto const& b = points_[i];
    auto const& c = points_[i_next];

    auto cross_product = (b[0] - a[0]) * (c[1] - a[1])
                         - (b[1] - a[1]) * (c[0] - a[0]);

    return cross_product <= 0 || soft_zero(cross_product);
}

//---------------------------------------------------------------------------//
/*!
 * Determine the next index using modular indexing.
 */
template<class T>
size_type ConvexHullFinder<T>::calc_next(size_type i) const
{
    return (i + 1) % points_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Determine the previous index using modular indexing.
 */
template<class T>
size_type ConvexHullFinder<T>::calc_previous(size_type i) const
{
    return (points_.size() + i - 1) % points_.size();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
