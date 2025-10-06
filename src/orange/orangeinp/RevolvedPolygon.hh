//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/RevolvedPolygon.hh
//---------------------------------------------------------------------------//
#pragma once

#include "ObjectInterface.hh"
#include "Solid.hh"

#include "detail/VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*! An azimuthally sliced arbitrary polygon revolved around the \em z axis.
 *
 * The polygon must be specified in counterclockwise order and may not be self
 * intersecting. The polygon cannot cross the \em z axis, i.e., all vertices
 * must satisfy \em r >= 0.
 *
 * Construction is performed using a convex differences tree approach
 * \citep{tor-convexdecomp-1984, https://doi.org/10.1145/357346.357348}. The
 * convex hull of the polygon is first found and revolved around the \em z
 * axis. Regions that constitute the difference between the convex hull and the
 * original polygon are then subtracted. Each of these regions is created
 * recursively in the same fashion. The recursion depth is referred to as the
 * "level" and each contiguous region within a level is a "region", as shown
 * below:
 * \verbatim
   original polygon         convex hull          difference
     |___     ____         |____________       |   ______
   ^ |   \    |  |         |           |       |   \    |  level 1
   | |     \  |  |         |           |       |     \  |  region 0
   z |       \|  |         | level 0   |       |       \|
     |           |    =    | region 0  |   -   |
   a |           |         |           |       |
   x |    /\     |         |           |       |     /\     level 1
   i |___/  \____|         |___________|       |    /__\    region 1
   s |_____________        |_____________      |_____________
      r axis ->
   \endverbatim
 * Convex "regions" are constructed from "subregions", as shown below:
 * \verbatim
     |   ______             |________                     |___
   ^ |   \    |  level 1    |        | level 1            |   \     level 1
   | |     \  |  region 0   |        | region 0           |     \   region 0
   z |       \|             |________| subregion 0        |_______\ subregion 1
     |                 =    |          (a cylinder)  -    |         (a cone)
   a |                      |                             |
   x |                      |                             |
   i |                      |                             |
   s |_____________         |_____________                |_____________
        r axis ->
   \endverbatim
 * In this example, level 1 region 0 is formed from only two subregions, but
 * the general case is handled via:
 *
 * region = union(outer subregions) - union(inner subregions).
 *
 * The final step in construction is azimuthal truncation, which is done
 * through a union operation with a negated or non-negated EnclosedAzi.
 *
 * \internal When labeling nodes in the CSG output, the following shorthand
 * format is used: `label@level.region.subregion`. For example, the final
 * subregion in the example above might be named `my_shape@1.0.1`. For each
 * level, additional nodes are created in the form: `label@level.suffix` where
 * suffixes have the following meanings:
 *
 *  1) .cu : the union of all concave regions on the level,
 *  2) .ncu : the negation of .cu,
 *  3) .d : the difference between the level's convex hull and .cu.
 *
 * For each region, additional nodes are created in the form
 * label@level.region.suffix where suffixes have the following meanings:
 *
 * 1) .ou : the union of nodes that comprise the outer boundary of the region,
 * 2) .iu : the union of nodes that comprise the inner boundary of the region,
 * 3) .nui : the negation of .ui,
 * 4) .d : the difference between .ou and .iu.
 *
 * If the supplied EnclosedAzi object is not [0, 2pi], additional nodes with
 * the following extensions are added:
 *
 * 1) azi/~azi : the enclosed, possibly negated, azimuthal angle,
 * 2) restricted : the intersection of the revolved polygon and azi/~azi.
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
    RevolvedPolygon(std::string&& label,
                    VecReal2&& polygon,
                    EnclosedAzi&& enclosed);

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

    //! Get the azimuthal angular restriction
    EnclosedAzi const& enclosed_azi() const { return enclosed_; }

  private:
    /// TYPES ///

    // Helper struct for keeping track of levels/regions/subregions
    struct SubIndex
    {
        size_type level = 0;
        size_type region = 0;
        size_type subregion = 0;
    };

    //// HELPER METHODS ////

    // Recursively construct convex regions, subtracting out concavities
    NodeId make_levels(detail::VolumeBuilder& vb,
                       VecReal2 const& polygon,
                       SubIndex si) const;

    // Revolve a convex polygon around the \em z axis
    NodeId make_region(detail::VolumeBuilder& vb,
                       VecReal2 const& polygon,
                       SubIndex si) const;

    // Make a translated cylinder node
    NodeId make_cylinder(detail::VolumeBuilder& vb,
                         Real2 const& p0,
                         Real2 const& p1,
                         SubIndex const& si) const;

    // Make a translated cone node
    NodeId make_cone(detail::VolumeBuilder& vb,
                     Real2 const& p0,
                     Real2 const& p1,
                     SubIndex const& si) const;

    // Make a label extension for a level
    std::string make_level_ext(SubIndex si) const;

    // Make a label extension for a region within a level
    std::string make_region_ext(SubIndex si) const;

    // Make a label extension for a subregion within a region
    std::string make_subregion_ext(SubIndex si) const;

    //// DATA ////

    std::string label_;
    VecReal2 polygon_;
    EnclosedAzi enclosed_;
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
