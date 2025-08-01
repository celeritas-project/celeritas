//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/RevolvedPolygon.hh
//---------------------------------------------------------------------------//
#pragma once

#include "ObjectInterface.hh"

#include "detail/VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*! An arbitrary (possibly concave) polygon revolved around the \em z axis.
 *
 * The polygon must be specified in counterclockwise order. Construction is
 * performed using a convex decomposition approach
 * \citep{tor-convexdecomp-1984, https://doi.org/10.1145/357346.357348}. The
 * convex hull of the polygon is first found and revolved around the \em z
 * axis. Regions that constitute the difference between the polygon and its
 * convex hull are then subtracted. Each of these regions is created
 * recursively in the same fashion. Because this method creates many regions,
 * these are kept track of using three indices for debugging purposes: the
level
 * index denotes the current recursion depth,


and the region index denotes the region within a given level.


An example of these indices is shown below.
 * Consider the following polygon:
 * \verbatim
     |            __________
   ^ |           |          |
   | |        ___|          |
   z |       |              |
     |       |              |
   a |       |              |
   x |       |            __|
   i |       |           |
   s |       |___________|
   \endverbatim
 * The convex hull of this polygon is used to create the first region:
 * \verbatim
     |            __________
   ^ |          /           |
   | |        /             |
   z |       |   level 0    |
     |       |   region  0  |
   a |       |              |
   x |       |              |
   i |       |             /
   s |       |___________/
   \endverbatim
 * Recursing one level deeper, we create two additional regions:
 * \verbatim
     |            ...........
   ^ |          /|  level 1, region 0
   | |        /__|          .
   z |       .              .
     |       .              .
   a |       .              .
   x |       .           ___.
   i |       .          |  /  level 1, region 1
   s |       ...........|/
   \endverbatim
 * and subtract their union from the first region.
 *
 * \internal When labeing nodes in the CSG output, the following shorthand
 *   format is used: `label@level.region`. For example, the final region
 *   in the example above might be named `my_shape@1.1`. For each level,
 *   additional nodes are created in the form: `label@level.suffix` where
 *   suffixes have the following meanings:
 *   1) .cu : the union of all convex regions on this level,
 *   2) .ncu : the negation of the union of all convex regions on this level,
 *   3) .d : the difference between this level's convex hull and the convex
 *      regions on this level.
 */
class RevolvedPolygon final : public ObjectInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstObject = std::shared_ptr<ObjectInterface const>;
    using VecReal2 = std::vector<Real2>;

    //!@}

    // Construct from a polygon
    RevolvedPolygon(std::string&& label, VecReal2&& polygon);

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

  private:
    /// TYPES ///

    // Helper struct for keeping track of embedded regions
    struct SubregionIndex
    {
        size_type level = 0;
        size_type region = 0;
        size_type subregion = 0;
    };

    //// HELPER METHODS ////

    // Recursively construct convex regions, subtracting out concavities
    NodeId make_levels(detail::VolumeBuilder& vb,
                       VecReal2 const& polygon,
                       SubregionIndex si) const;

    // Revolved a *convex* polygon around the \em z axis
    NodeId make_region(detail::VolumeBuilder& vb,
                       VecReal2 const& polygon,
                       SubregionIndex si) const;

    // Make a translated cylinder node
    NodeId make_cylinder(detail::VolumeBuilder& vb,
                         Real2 const& p0,
                         Real2 const& p1,
                         SubregionIndex const& si) const;

    // Make a translated cone node
    NodeId make_cone(detail::VolumeBuilder& vb,
                     Real2 const& p0,
                     Real2 const& p1,
                     SubregionIndex const& si) const;

    // Make a label for a level
    std::string make_level_ext(SubregionIndex si) const;

    // Make a label for a region within a level
    std::string make_region_ext(SubregionIndex si) const;

    // Make a label for a subregion within a region
    std::string make_subregion_ext(SubregionIndex si) const;

    //// DATA ////

    std::string label_;
    VecReal2 polygon_;
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
