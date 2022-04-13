//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SurfaceFunctors.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Algorithms.hh"
#include "base/Array.hh"
#include "base/Assert.hh"

#include "Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Calculate the sense of a surface at a given position.
struct CalcSense
{
    const Real3& pos;

    template<class S>
    CELER_FUNCTION SignedSense operator()(S&& surf)
    {
        return surf.calc_sense(this->pos);
    }
};

//---------------------------------------------------------------------------//
//! Get the number of intersections of a surface
template<class S>
struct NumIntersections
{
    CELER_CONSTEXPR_FUNCTION size_type operator()() const noexcept
    {
        return typename S::Intersections{}.size();
    }
};

//---------------------------------------------------------------------------//
//! Calculate the outward normal at a position.
struct CalcNormal
{
    const Real3& pos;

    template<class S>
    CELER_FUNCTION Real3 operator()(S&& surf)
    {
        return surf.calc_normal(this->pos);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Calculate the smallest distance from a point to the surface.
 *
 * For certain surface types (spheres, cylinders, planes), defined such that
 * the normal is *outward* (positive when "outside", negative when "inside"),
 * the nearest distance to the surface can be calculated quite trivially.
 */
struct CalcSafetyDistance
{
    const Real3& pos;

    //! Operate on a surface
    template<class S>
    CELER_FUNCTION real_type operator()(S&& surf)
    {
        if (!S::simple_safety())
        {
            // Not a surface that satisfies our simplifying constraints: return
            // a conservative answer.
            return 0;
        }

        // Calculate outward normal
        Real3 dir = surf.calc_normal(this->pos);

        auto sense = surf.calc_sense(this->pos);
        // If sense is "positive" (on or outside), flip direction to inward so
        // that the vector points toward the surface
        if (sense == SignedSense::outside)
        {
            for (real_type& d : dir)
            {
                d *= -1;
            }
        }
        else if (sense == SignedSense::on)
        {
            return 0;
        }

        // Return the closest intersection
        auto intersect
            = surf.calc_intersections(this->pos, dir, SurfaceState::off);
        return *celeritas::min_element(intersect.begin(), intersect.end());
    }
};

//---------------------------------------------------------------------------//
/*!
 * Fill an array with valid distances-to-intersection.
 *
 * This assumes that each call is to the next face index, starting with face
 * zero.
 */
template<class IsValid>
class CalcIntersections
{
  public:
    //! Construct from the particle point, direction, face ID, and temp storage
    CELER_FUNCTION CalcIntersections(const Real3&        pos,
                                     const Real3&        dir,
                                     IsValid             is_valid_isect,
                                     FaceId              on_face,
                                     bool                is_simple,
                                     const TempNextFace& next_face)
        : pos_(pos)
        , dir_(dir)
        , is_valid_isect_(is_valid_isect)
        , on_face_idx_(on_face.unchecked_get())
        , fill_isect_(!is_simple)
        , face_(next_face.face)
        , distance_(next_face.distance)
        , isect_(next_face.isect)
    {
        CELER_EXPECT(face_ && distance_);
    }

    //! Operate on a surface
    template<class S>
    CELER_FUNCTION void operator()(S&& surf)
    {
        auto on_surface = (on_face_idx_ == face_idx_) ? SurfaceState::on
                                                      : SurfaceState::off;

        // Calculate distance to surface along this direction
        auto all_dist = surf.calc_intersections(pos_, dir_, on_surface);

        // Copy possible intersections and this surface to the output
        for (real_type dist : all_dist)
        {
            CELER_ASSERT(dist > 0);
            if (is_valid_isect_(dist))
            {
                // Save intersection in the list
                face_[isect_idx_]     = FaceId{face_idx_};
                distance_[isect_idx_] = dist;
                if (fill_isect_)
                {
                    isect_[isect_idx_] = isect_idx_;
                }
                ++isect_idx_;
            }
        }
        // Increment to next face
        ++face_idx_;
    }

    CELER_FUNCTION size_type face_idx() const { return face_idx_; }
    CELER_FUNCTION size_type isect_idx() const { return isect_idx_; }

  private:
    //// DATA ////

    const Real3&     pos_;
    const Real3&     dir_;
    const IsValid    is_valid_isect_;
    const size_type  on_face_idx_;
    const bool       fill_isect_;
    FaceId* const    face_;
    real_type* const distance_;
    size_type* const isect_;
    size_type        face_idx_{0};
    size_type        isect_idx_{0};
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
