//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/ConvexHullFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <numeric>
#include <type_traits>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * SFINAE struct for determing if a templated type can is a valid Real2
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
 * Find the convex hull of ordered, 2D points with the Graham Scan algorithm.
 *
 * Here, "ordered" means:
 * 1) Segments between adjacent points, including the first and last
 *    points, comprise a non-self-intersecting polygon,
 * 2) Points are supplied in a clockwise order.
 *
 * The Gram Scan exploits this ordered configuration to evaluate with
 * O(N log(N)) time complexity.
 */
template<class Real2>
class ConvexHullFinder
{
    static_assert(is_valid_real2<Real2>::value,
                  "Real2 must support contain two arithmatic types accessibly "
                  "by indexing");

  public:
    //!@{
    //! \name Type aliases
    using Points = std::vector<Real2>;
    using index_type = size_type;
    using ConvexMask = std::vector<bool>;
    using PointSpan = Span<Real2 const>;
    using ConcaveRegions = std::vector<PointSpan>;

    struct Results
    {
        ConvexMask convex_mask;
        ConcaveRegions concave_regions;
    };
    //!@}

  public:
    // Contruct with vector of ordered points
    explicit ConvexHullFinder(Points& points);

    // Find the convex hull for the points within PointSpan
    Results operator()(PointSpan points);

  private:
    using Index3 = std::array<index_type, 3>;

    struct NextIndexCalculator
    {
      public:
        NextIndexCalculator(index_type size) : size_(size) {};
        index_type operator()(index_type i) { return (i + 1) % size_; };

      private:
        index_type size_;
    };

  private:
    Points const& all_points_;

  private:
    index_type min_element_idx(Span<Real2 const> const& points) const;
    bool
    is_clockwise(Span<Real2 const> const& points, Index3 const& indices) const;

    ConcaveRegions find_concave_regions(PointSpan const& points,
                                        ConvexMask const& mask) const;
};

//---------------------------------------------------------------------------//
/*!
 * Contruct with ordered points.
 */
template<class Real2>
ConvexHullFinder<Real2>::ConvexHullFinder(ConvexHullFinder::Points& points)
    : all_points_{points}
{
    CELER_EXPECT(all_points_.size() >= 2);
}

//---------------------------------------------------------------------------//
/*!
 * Find the convex hull for the points within the supplied span.
 */
template<class Real2>
auto ConvexHullFinder<Real2>::operator()(PointSpan points) -> Results
{
    Results results;
    auto n = points.size();

    // Return early for trivial cases
    if (n < 4)
    {
        results.convex_mask = ConvexMask(n, true);
        return results;
    }

    // Find the indices of the points on the convex hull
    std::vector<index_type> hull;
    {
        NextIndexCalculator calc_next(n);
        index_type i = this->min_element_idx(points);
        hull.push_back(i);
        i = calc_next(i);

        for ([[maybe_unused]] auto _ : range(n - 1))
        {
            index_type i_next = calc_next(i);

            if (this->is_clockwise(points, {hull.back(), i, i_next}))
            {
                hull.push_back(i);
            }
            else
            {
                while (!this->is_clockwise(
                    points, {hull[hull.size() - 2], hull.back(), i_next}))
                {
                    hull.pop_back();
                }
            }

            i = i_next;
        }
    }

    // Convert convex hull indices to a mask
    results.convex_mask = ConvexMask(n, false);
    for (auto h : hull)
    {
        results.convex_mask[h] = true;
    }

    // Add concave regions
    results.concave_regions
        = this->find_concave_regions(points, results.convex_mask);

    return results;
}

//---------------------------------------------------------------------------//
// >> HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Find the index of the element with the lowest y value.
 */
template<class Real2>
auto ConvexHullFinder<Real2>::min_element_idx(Span<Real2 const> const& points) const
    -> index_type
{
    auto starting_it = std::min_element(
        points.begin(), points.end(), [](Real2 const& a, Real2 const& b) {
            return a[1] < b[1];
        });
    return std::distance(points.begin(), starting_it);
};
//---------------------------------------------------------------------------//
/*!
 * Determine if three elements form a clockwise turn using the cross product.
 *
 * Colinear points are counted as clockwise.
 */
template<class Real2>
auto ConvexHullFinder<Real2>::is_clockwise(
    Span<Real2 const> const& points,
    ConvexHullFinder<Real2>::Index3 const& indices) const -> bool
{
    auto const& a = points[indices[0]];
    auto const& b = points[indices[1]];
    auto const& c = points[indices[2]];

    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0;
}

//---------------------------------------------------------------------------//
/*!
 * Find the concave regions
 */
template<class Real2>
auto ConvexHullFinder<Real2>::find_concave_regions(
    Span<Real2 const> const& points, ConvexMask const& mask) const
    -> ConcaveRegions
{
    ConcaveRegions regions;
    NextIndexCalculator calc_next(points.size());

    // Start at 1, because the first point is gaurenteed to be on the hull
    size_type i = 1;
    while (i != 0)
    {
        if (mask[i])
        {
            i = calc_next(i);
        }
        else
        {
            index_type start = i - 1;
            do
            {
                i = calc_next(i);
            } while (!mask[i]);

            regions.push_back(PointSpan(&points[start], &points[i + 1]));
        }
    }
    return regions;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
