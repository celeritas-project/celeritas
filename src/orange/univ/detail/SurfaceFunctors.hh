//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/SurfaceFunctors.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArraySoftUnit.hh"
#include "corecel/math/NumericLimits.hh"
#include "orange/surf/detail/IntersectionSolver.hh"

#include "Types.hh"
#include "Utils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Calculate the sense of a surface at a given position.
struct CalcSense
{
    Real3 const& pos;

    template<class S>
    CELER_FUNCTION SignedSense operator()(S const& surf)
    {
        return surf.calc_sense(this->pos);
    }
};

//---------------------------------------------------------------------------//
//! Get the number of intersections of a surface
struct NumIntersections
{
    template<class ST>
    CELER_CONSTEXPR_FUNCTION size_type operator()(ST) const noexcept
    {
        using S = typename ST::type;
        return typename S::Intersections{}.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Calculate the smallest distance from a point to the surface.
 *
 * Surface types that support "simple safety" (spheres, cylinders, planes,
 * cones) provide the exact distance to their nearest point in closed form. All
 * other surfaces return a conservative distance of zero.
 */
struct CalcSafetyDistance
{
    Real3 const& pos;

    //! Operate on a surface
    template<class S>
    CELER_FUNCTION real_type operator()(S const& surf)
    {
        if constexpr (S::simple_safety())
        {
            return surf.calc_safety(this->pos);
        }
        else
        {
            return 0;
        }
    }
};

//---------------------------------------------------------------------------//
/*!
 * Fill an array with valid distances-to-intersection.
 *
 * \tparam F Predicate for returning whether the distance is allowable
 *
 * This assumes that each call is to the next face index, starting with face
 * zero.
 */
class CalcIntersections
{
  public:
    //! Construct from the particle point, direction, face ID, and temp storage
    CELER_FUNCTION CalcIntersections(real_type max_dist,
                                     Real3 const& pos,
                                     Real3 const& dir,
                                     FaceId on_face,
                                     bool is_simple,
                                     TempNextFace const& next_face)
        : is_valid_isect_(max_dist)
        , pos_(pos)
        , dir_(dir)
        , on_face_idx_(on_face.unchecked_get())
        , face_(next_face.face)
        , distance_(next_face.distance)
        , isect_(!is_simple ? next_face.isect : nullptr)
    {
        CELER_EXPECT(face_ && distance_);
    }

    //! Operate on a surface
    template<class S>
    CELER_FUNCTION void operator()(S const& surf)
    {
        using Surface = S;
        auto on_surface = (on_face_idx_ == face_idx_) ? SurfaceState::on
                                                      : SurfaceState::off;
        if constexpr (Surface::surface_type() == SurfaceType::inv)
        {
            CELER_ASSERT_UNREACHABLE();
            return;
        }
        else if constexpr (surface_type_group(Surface::surface_type())
                           == SurfaceTypeGroup::planar)
        {
            if (on_surface == SurfaceState::off)
            {
                // Can only intersect if not on the surface
                this->solve_and_store(
                    surf.calc_intersection_planar(pos_, dir_, on_surface));
            }
        }
        else if constexpr (surface_type_group(Surface::surface_type())
                           == SurfaceTypeGroup::quadric)
        {
            this->solve_and_store(
                surf.calc_intersection_quadratic(pos_, dir_, on_surface));
        }
        else
        {
            this->solve_and_store(
                surf.calc_intersection_quartic(pos_, dir_, on_surface));
        }
        // Increment to next face
        ++face_idx_;
    }

    CELER_FUNCTION size_type face_idx() const { return face_idx_; }
    CELER_FUNCTION size_type isect_idx() const { return isect_idx_; }

  private:
    //// DATA ////

    IsNotFurtherThan is_valid_isect_;
    Real3 const& pos_;
    Real3 const& dir_;
    size_type const on_face_idx_;
    FaceId* const face_;
    real_type* const distance_;
    size_type* const isect_;
    FaceId::size_type face_idx_{0};
    size_type isect_idx_{0};

    //// HELPERS ////

    template<class T>
    CELER_FUNCTION void solve_and_store(T const& input)
    {
        return this->store_intersections(detail::solve_intersections(input));
    }

    template<size_type N>
    CELER_FUNCTION void store_intersections(Array<real_type, N> const& all_dist)
    {
        for (real_type dist : all_dist)
        {
            this->store_intersection(dist);
        }
    }

    CELER_FUNCTION void store_intersection(real_type dist)
    {
        CELER_EXPECT(dist > 0);
        if (is_valid_isect_(dist))
        {
            face_[isect_idx_] = FaceId{face_idx_};
            distance_[isect_idx_] = dist;
            if (isect_)
            {
                isect_[isect_idx_] = isect_idx_;
            }
            ++isect_idx_;
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
