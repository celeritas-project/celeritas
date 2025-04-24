//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/ConvexHullFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "orange/OrangeTypes.hh"
#include "orange/univ/detail/Utils.hh"

#include "PolygonUtils.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Find the convex hull of a sequence of 2D points.
 *
 * This helper class does not take ownership of the supplied points and
 * tolerance, so the lifetime of objects of this class should be shorter than
 * the lifetime of these arguments. Points must be supplied in clockwise-order
 * such that segments between adjacent points, including the last and first
 * points, comprise a non-self-intersecting polygon. Exploiting this ordering,
 * the Graham Scan algorithm \citet{graham-convexhull-1972,
 * https://doi.org/10.1016/0020-0190(72)90045-2} finds the convex hull with
 * O(N) time complexity.
 */
template<class T>
class ConvexHullFinder
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = T;
    using Real2 = celeritas::Array<real_type, 2>;
    using VecReal2 = std::vector<Real2>;
    using VecVecReal2 = std::vector<VecReal2>;
    //!@}

  public:
    // Construct with vector of ordered points and a tolerance
    explicit ConvexHullFinder(VecReal2 const& points, Tolerance<> const& tol);

    // Make the convex hull
    VecReal2 make_convex_hull() const;

    // Calculate the concave regions, each supplied in clockwise order
    VecVecReal2 calc_concave_regions() const;

  private:
    /// TYPES ///
    using ConvexMask = std::vector<bool>;

    /// DATA ///
    VecReal2 const& points_;
    Tolerance<> const& tol_;
    SoftOrientation<real_type> soft_ori_;
    ConvexMask convex_mask_;
    size_type start_index_;

    /// HELPER FUNCTIONS ///

    // Make a SoftOrientation object based on the tolerance and polygon extents
    SoftOrientation<real_type> make_soft_ori() const;

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
ConvexHullFinder<T>::ConvexHullFinder(ConvexHullFinder::VecReal2 const& points,
                                      Tolerance<> const& tol)
    : points_{points}, tol_{tol}
{
    CELER_EXPECT(points_.size() > 2);

    soft_ori_ = this->make_soft_ori();
    start_index_ = this->min_element_idx();
    convex_mask_ = this->calc_convex_mask();
}

//---------------------------------------------------------------------------//
/*!
 * Make a SoftOrientation object based on the tolerance and polygon extents.
 */
template<class T>
auto ConvexHullFinder<T>::make_soft_ori() const -> SoftOrientation<real_type>
{
    auto const x_cmp
        = [](Real2 const& a, Real2 const& b) { return a[0] < b[0]; };
    auto const y_cmp
        = [](Real2 const& a, Real2 const& b) { return a[1] < b[1]; };

    auto const [x_min, x_max]
        = std::minmax_element(points_.begin(), points_.end(), x_cmp);
    auto const [y_min, y_max]
        = std::minmax_element(points_.begin(), points_.end(), y_cmp);

    // Convert min/max x and y values to extents
    using extent_type = Real3::value_type;
    Real3 const extents{static_cast<extent_type>((*x_max)[0] - (*x_min)[0]),
                        static_cast<extent_type>((*y_max)[1] - (*y_min)[1]),
                        0};

    return SoftOrientation<real_type>(
        ::celeritas::detail::BumpCalculator(tol_)(extents));
}

//---------------------------------------------------------------------------//
/*!
 * Make the convex hull.
 */
template<class T>
auto ConvexHullFinder<T>::make_convex_hull() const -> VecReal2
{
    CELER_EXPECT(convex_mask_.size() > 2);

    VecReal2 convex_hull;
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
auto ConvexHullFinder<T>::calc_concave_regions() const -> VecVecReal2
{
    CELER_EXPECT(convex_mask_.size() > 2);
    VecVecReal2 concave_regions;

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
            VecReal2 concave_region;
            concave_region.push_back(points_[calc_next(i)]);
            do
            {
                concave_region.push_back(points_[i]);
                i = this->calc_previous(i);
            } while (!convex_mask_[i]);

            concave_region.push_back(points_[i]);
            concave_regions.push_back(std::move(concave_region));
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
 * This method uses the Graham Scan algorithm.
 */
template<class T>
auto ConvexHullFinder<T>::calc_convex_mask() const -> ConvexMask
{
    // Find the indices of the points on the convex hull. Start from the point
    // with the lowest y value, which is guaranteed to be on the hull.
    std::vector<size_type> hull;
    auto i = start_index_;
    hull.push_back(i);
    i = this->calc_next(i);

    for (size_type j = 0, n = points_.size(); j != n; ++j)
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
        points_.begin(), points_.end(), [](Real2 const& a, Real2 const& b) {
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
    return soft_ori_(a, b, c) != detail::Orientation::counterclockwise;
}

//---------------------------------------------------------------------------//
/*!
 * Determine the next index using modular indexing.
 */
template<class T>
size_type ConvexHullFinder<T>::calc_next(size_type i) const
{
    return (i + 1 < points_.size() ? i + 1 : 0);
}

//---------------------------------------------------------------------------//
/*!
 * Determine the previous index using modular indexing.
 */
template<class T>
size_type ConvexHullFinder<T>::calc_previous(size_type i) const
{
    return (i > 0 ? i : points_.size()) - 1;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
