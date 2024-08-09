//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/OrientedBoundingZone.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/OrangeData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Oriented bounding zone for saftey distance calculations.
 *
 * Here, the bounding zone is defined by an inner and outer bounding box
 * (specified via half-widths) transformed by the same transformation;
 * pressumably the transformation of the volume they correspond to. The
 * safety distance can be approximated as the minimum distance to the inner or
 * outer box. For points lying between the inner and outer boxes, the safety
 * distance is zero.
 */
class OrientedBoundingZone
{
  public:
    //!@{
    //! \name Type aliases
    using Storage = NativeCRef<OrientedBoundingZoneData>;
    //!@}

  public:
    // Construct from inner/outer half-widths and a corresponding transform
    OrientedBoundingZone(Real3Id inner_hw_id,
                         Real3Id outer_hw_id,
                         TransformId transform_id,
                         Storage* storage);

    // Calculate the safety distance for any position inside the outer box
    real_type safety_distance_inside(Real3 pos);

    // Calculate the safety distance for any position outside the inner box
    real_type safety_distance_outside(Real3 pos);

    // Determine if a position is inside the inner box
    bool is_inside_inner(Real3 const& pos);

    // Determine if a position is inside the outer box
    bool is_inside_outer(Real3 const& pos);

  private:
    // >> DATA
    Real3Id inner_hw_id_;
    Real3Id outer_hw_id_;
    TransformId transform_id_;
    Storage* storage_;

    //// HELPER METHODS ////

    // Translate a position into the OBZ coordinate system
    Real3 translate(Real3 const& pos);

    // Reflect a position into quadrant zero
    Real3 quadrant_zero(Real3 const& pos);

    // Determine if a translated position is inside a translated box
    bool is_inside(Real3 const& trans_pos, Real3 const& half_widths);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
