//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/StackedExtrudedPolygon.hh
//---------------------------------------------------------------------------//
#pragma once

#include "ObjectInterface.hh"

#include "detail/VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*! A convex/concave polygon extruded along a polyline, with scaling.
 *
 * The polygon must be specified in counterclockwise order. The polyline must
 * be strictly monotonically increasing in z. Scaling factors can be any
 * positive value. Scaling is assumed to occur with respect to the polygon's
 * original coordinate system.
 *
 * Construction is performed using a convex decomposition approach
 * \citep{tor-convexdecomp-1984, https://doi.org/10.1145/357346.357348}. The
 * convex hull of the polygon is first found, then extruded along the polyline
 * (with scaling) form a stack of ExtrudedPolygon shapes. Regions that
 * constitute the difference between the polygon and its convex hull are then
 * subtracted. Each of these regions is created recursively in the same
 * fashion, i.e. finding the convex hull and creating a stack.
 *
 * Because this method creates many regions, these are kept track of using
 * three indices for debugging purposes: level, stack, and segment. The level
 * index denotes the current recursion depth. The stack index denotes the
 * convex region which is extruded into a stack on a given level. The segment
 * index denotes the z segment within the stack. An example of these indices is
 * shown below. Consider the following polygon, extruded along a single-segment
 * polyline:
 * \verbatim
             __________
            |          |
         ___|          |
        |              |
        |              |
        |              |
        |            __|
        |           |
        |___________|
   \endverbatim
 * The convex hull of this polygon is used to create the first region:
 * \verbatim
             __________
           /           |
         /             |
        |   level 0    |
        |   stack 0    |
        |   segment 0  |
        |              |
        |             /
        |___________/
   \endverbatim
 * Recursing one level deeper, we create two additional regions:
 * \verbatim

             ...........
           /|  level 1, stack 0, segment 0
         /__|          .
        .              .
        .              .
        .              .
        .           ___.
        .          |  /  level 1, stack 1, segment 0
        ...........|/
   \endverbatim
 * and subtract their union from the first region.
 *
 * \internal When labeing nodes in the CSG output, the following shorthand
 *   format is used: `label@level.stack.segment`. For example, the final region
 *   in the example above might be named `my_shape@1.1.0`. For each level,
 *   additional nodes are created in the form: `label@level.suffix` where
 *   suffixes have the following meanings:
 *   1) .cu : the union of all convex regions on this level,
 *   2) .ncu : the negation of the union of all convex regions on this level,
 *   3) .d : the difference between this level's convex hull and the convex
 *      regions on this level.
 */
class StackedExtrudedPolygon final : public ObjectInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstObject = std::shared_ptr<ObjectInterface const>;
    using VecReal = std::vector<real_type>;
    using VecReal2 = std::vector<Real2>;
    using VecReal3 = std::vector<Real3>;
    //!@}

    //! Construct, or return an ExtrudedPolygon shape if possible
    static SPConstObject or_solid(std::string&& label,
                                  VecReal2&& polygon,
                                  VecReal3&& polyline,
                                  VecReal&& scaling);

    // Construct from a polygon, polyline, and scaling factors
    StackedExtrudedPolygon(std::string&& label,
                           VecReal2&& polygon,
                           VecReal3&& polyline,
                           VecReal&& scaling);

    //// INTERFACE ////

    //! Get the user-provided label
    std::string_view label() const final { return label_; }

    //! Construct a volume from this object
    NodeId build(VolumeBuilder&) const final;

    //! Write the shape to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Get the polygon
    VecReal2 const& polygon() const { return polygon_; };

    //! Get the polyline
    VecReal3 const& polyline() const { return polyline_; };

    //! Get the scaling factors
    VecReal const& scaling() const { return scaling_; };

  private:
    /// TYPES ///

    // Helper struct for keeping track of embedded regions
    struct SubRegionIndex
    {
        size_type level = 0;
        size_type stack = 0;
    };

    //// HELPER METHODS ////

    // Recursively construct stacks, subtracting out concavities
    NodeId make_levels(detail::VolumeBuilder& vb,
                       VecReal2 const& polygon,
                       SubRegionIndex si) const;

    // Extrude a *convex* polygon along the polyline
    NodeId make_stack(detail::VolumeBuilder& vb,
                      VecReal2 const& polygon,
                      SubRegionIndex si) const;

    // Make a label for a level
    std::string make_level_label(SubRegionIndex si) const;

    // Make a label for a stack within a level
    std::string make_stack_label(SubRegionIndex si) const;

    // Make a label for a segment within a stack
    std::string
    make_segment_label(SubRegionIndex si, size_type segment_idx) const;

    //// DATA ////

    std::string label_;
    VecReal2 polygon_;
    VecReal3 polyline_;
    VecReal scaling_;
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
