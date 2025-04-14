//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/ConvexHullFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/SoftEqual.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * SFINAE struct for determing if a templated type is a valid Real2.
 */
template<typename T>
struct is_valid_real2
{
  private:
    //! Matches if T is a valid Real2
    template<typename U>
    static auto match_real2(int)
        -> decltype(std::is_arithmetic<decltype(std::declval<U>()[0])>{},
                    std::is_arithmetic<decltype(std::declval<U>()[1])>{},
                    std::true_type{});

    //! Fallback; always matches
    template<typename>
    static auto match_real2(...) -> std::false_type;

  public:
    static constexpr bool value = decltype(match_real2<T>(0))::value;
};

}  // namespace
//---------------------------------------------------------------------------//
/*!
 * Find the convex hull of a sequence of 2D points.
 *
 * These points must be supplied in clockwise-order such that segments between
 * adjacent points, including the last and first points, comprise a
 * non-self-intersecting polygon. Exploiting this ordering, the Gram Scan
 * algorithm finds the convex hull with O(N) time complexity.
 */
template<class Real2>
class ConvexHullFinder
{
    static_assert(is_valid_real2<Real2>::value,
                  "Real2 must support contain two arithmatic types accessible "
                  "via operator[].");

  public:
    //!@{
    //! \name Type aliases
    using Points = std::vector<Real2>;
    using ConcaveRegions = std::vector<Points>;
    //!@}

  public:
    // Contruct with vector of ordered points
    explicit ConvexHullFinder(Points const& points);

    // Return the convex hull
    Points convex_hull() const;

    // Calculate the concave regions, each supplied in clockwise order
    ConcaveRegions calc_concave_regions() const;

  private:
    /// TYPES ///
    using ConvexMask = std::vector<bool>;
    using index_type = size_type;
    using Index3 = std::array<index_type, 3>;

    /// DATA ///
    Points const& points_;
    ConvexMask convex_mask_;
    index_type start_index_;
    SoftZero<> soft_zero_;

    /// HELPER FUNCTIONS ///

    // Calculate a mask that indicates which points are on the convex hull
    void calc_convex_mask();

    // Find the index of the element with the minimum y value
    void min_element_idx();

    // Determine if three points, traversed in order, form a clockwise turn
    bool is_clockwise(Index3 const& indices) const;

    // Determine the next index, with modular indexing
    index_type calc_next(index_type i) const;

    // Determine the previous index, with modular indexing
    index_type calc_previous(index_type i) const;
};

//---------------------------------------------------------------------------//
/*!
 * Contruct with vector of ordered points.
 *
 * This function generates a mask that is used to calculate the convex hull
 * and associated concave regions. Note that this function does not enforce
 * ordering.
 */
template<class Real2>
ConvexHullFinder<Real2>::ConvexHullFinder(ConvexHullFinder::Points const& points)
    : points_{points}
{
    CELER_EXPECT(points_.size() > 2);
    this->min_element_idx();
    this->calc_convex_mask();
}

//---------------------------------------------------------------------------//
/*!
 * Return the convex hull.
 */
template<class Real2>
auto ConvexHullFinder<Real2>::convex_hull() const -> Points
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
template<class Real2>
auto ConvexHullFinder<Real2>::calc_concave_regions() const -> ConcaveRegions
{
    CELER_EXPECT(convex_mask_.size() > 2);
    ConcaveRegions concave_regions;

    // Since the original shape was supplied in clockwise order, we must
    // traverse the points backwards in order to obtain the concave regions in
    // clockwise order.
    index_type i = this->calc_previous(start_index_);
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
template<class Real2>
void ConvexHullFinder<Real2>::calc_convex_mask()
{
    auto n = points_.size();

    // Find the indices of the points on the convex hull. Start from the point
    // with the lowest y value, which is gaurenteed to be on the hull.
    std::vector<index_type> hull;
    auto i = start_index_;
    hull.push_back(i);
    i = this->calc_next(i);

    for ([[maybe_unused]] auto _ : range(n - 1))
    {
        index_type i_next = this->calc_next(i);

        if (this->is_clockwise({hull.back(), i, i_next}))
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
                       {hull[hull.size() - 2], hull.back(), i_next}))
            {
                hull.pop_back();
            }
        }

        i = i_next;
    }

    // Convert convex hull indices to a mask
    convex_mask_ = ConvexMask(n, false);
    for (auto h : hull)
    {
        convex_mask_[h] = true;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Find the index of the element with the lowest y value.
 */
template<class Real2>
void ConvexHullFinder<Real2>::min_element_idx()
{
    auto starting_it = std::min_element(
        points_.begin(), points_.end(), [](Real2 const& a, Real2 const& b) {
            return a[1] < b[1];
        });
    start_index_ = std::distance(points_.begin(), starting_it);
};

//---------------------------------------------------------------------------//
/*!
 * Determine if three elements form a clockwise turn using the cross product.
 *
 * Here, colinear points are considered clockwise.
 */
template<class Real2>
auto ConvexHullFinder<Real2>::is_clockwise(
    ConvexHullFinder<Real2>::Index3 const& indices) const -> bool
{
    auto const& a = points_[indices[0]];
    auto const& b = points_[indices[1]];
    auto const& c = points_[indices[2]];

    auto cross_product = (b[0] - a[0]) * (c[1] - a[1])
                         - (b[1] - a[1]) * (c[0] - a[0]);
    return cross_product <= 0 || soft_zero(cross_product);
}

//---------------------------------------------------------------------------//
/*!
 * Determine the next index using modular indexing.
 */
template<class Real2>
auto ConvexHullFinder<Real2>::calc_next(ConvexHullFinder<Real2>::index_type i) const
    -> index_type
{
    return (i + 1) % points_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Determine the previous index using modular indexing.
 */
template<class Real2>
auto ConvexHullFinder<Real2>::calc_previous(
    ConvexHullFinder<Real2>::index_type i) const -> index_type
{
    return (points_.size() + i - 1) % points_.size();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
