//------------------------------- -*- C++ -*- -------------------------------//
//  Copyright Celeritas contributors: see top-level COPYRIGHT file for details
//  SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/RevolvedPolygon.cc
//---------------------------------------------------------------------------//
#include "RevolvedPolygon.hh"

#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "orange/transform/Translation.hh"

#include "ObjectIO.json.hh"
#include "Shape.hh"

#include "detail/BuildIntersectRegion.hh"
#include "detail/ConvexHullFinder.hh"

namespace celeritas
{
namespace orangeinp
{
namespace
{
//! Convenience enumeration for implementations in this file
enum
{
    R = 0,
    Z = 1
};
}  // namespace
//---------------------------------------------------------------------------//
/*!
 * Construct from a polygon.
 */
RevolvedPolygon::RevolvedPolygon(std::string&& label, VecReal2&& polygon)
    : label_{std::move(label)}, polygon_{std::move(polygon)}
{
    CELER_VALIDATE(polygon_.size() >= 3,
                   << "polygon must have at least 3 vertices");

    // All points must be positive
    CELER_VALIDATE(std::all_of(polygon_.begin(),
                               polygon_.end(),
                               [](Real2 const& p) { return p[R] >= 0; }),
                   << "polygon must consist of only positive r values");
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume from this shape.
 */
NodeId RevolvedPolygon::build(VolumeBuilder& vb) const
{
    // Use the volume builder's tolerance to remove any colinear points
    auto filtered_polygon
        = detail::filter_collinear_points(polygon_, vb.tol().abs);

    // After removing collinear points, at least 3 points must remain
    CELER_VALIDATE(filtered_polygon.size() >= 3,
                   << "polygon must consist of at least 3 points");

    // Start the recursion process
    SubIndex ri;
    return this->make_levels(vb, filtered_polygon, ri);
}

//---------------------------------------------------------------------------//
/*!
 * Write the shape to JSON.
 */
void RevolvedPolygon::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}
//-------------------------------------------------------------------------//
// HELPER FUNCTION DEFINITIONS
//-------------------------------------------------------------------------//
/*!
 * Recursively construct convex regions, subtracting out concavities.
 */
NodeId RevolvedPolygon::make_levels(detail::VolumeBuilder& vb,
                                    VecReal2 const& polygon,
                                    RevolvedPolygon::SubIndex si) const
{
    // Find the convex hull
    detail::ConvexHullFinder<real_type> hull_finder(polygon, vb.tol());
    auto convex_hull = hull_finder.make_convex_hull();
    auto concave_regions = hull_finder.calc_concave_regions();

    // Build the convex region
    auto filtered_convex_hull
        = detail::filter_collinear_points(convex_hull, vb.tol().abs);
    NodeId result = this->make_region(vb, filtered_convex_hull, si);

    // Return early if there are no concave regions to process
    if (concave_regions.empty())
    {
        return result;
    }

    // Create a vector of all concave regions, via recursion
    std::vector<NodeId> concave_nodes(concave_regions.size());
    for (auto i : range(static_cast<size_type>(concave_nodes.size())))
    {
        concave_nodes[i] = this->make_levels(
            vb, concave_regions[i], SubIndex{si.level + 1, i});
    }

    auto level_ext = this->make_level_ext(si);

    // Create a union of all concave regions
    NodeId concave_union
        = vb.insert_region(Label{label_, level_ext + ".cu"},
                           Joined{op_or, std::move(concave_nodes)});

    // Create a negation of this union
    auto sub_node = vb.insert_region(Label{label_, level_ext + ".ncu"},
                                     Negated{concave_union});

    // Subtract concave regions from the convex hull
    return vb.insert_region(Label{label_, level_ext + ".d"},
                            Joined{op_and, {result, sub_node}});
}

//---------------------------------------------------------------------------//
/*!
 * Revolved a convex polygon around the \em z axis.
 *
 * The polygon must be strictly counterclockwise, i.e., it must not contain any
 * collinear points. Construction is done by creating a set of outer subregions
 * that define the outer edge of the polygon and subtracting out a set of inner
 * subregions.
 */
NodeId RevolvedPolygon::make_region(detail::VolumeBuilder& vb,
                                    VecReal2 const& polygon,
                                    SubIndex si) const
{
    // The polygon should have a strictly clockwise orientation
    CELER_VALIDATE(has_orientation(make_span(polygon),
                                   detail::Orientation::counterclockwise),
                   << "polygon must be specified in strictly counterclockwise "
                      "order");

    SoftEqual<real_type> soft_equal(vb.tol().rel, vb.tol().abs);
    SoftZero<real_type> soft_zero(vb.tol().abs);

    auto n = polygon.size();
    std::vector<NodeId> outer_nodes;
    std::vector<NodeId> inner_nodes;

    // Create subregions for each pair of adjacent points that do not form
    // a horizontal line or a line that coincides with the z axis
    for (auto i : range(n))
    {
        auto const& p0 = polygon[i];
        auto const& p1 = polygon[(i + 1) % n];

        if (soft_equal(p0[Z], p1[Z]) || (soft_zero(p0[R]) && soft_zero(p1[R])))
        {
            // Perpendicular to or coincide with z, don't make a shape
            continue;
        }

        // Make a cylinder or cone, and add it to the inner/outer nodes vector
        NodeId shape_id = soft_equal(p0[R], p1[R])
                              ? this->make_cylinder(vb, p0, p1, si)
                              : this->make_cone(vb, p0, p1, si);
        (p0[Z] < p1[Z] ? outer_nodes : inner_nodes).push_back(shape_id);

        si.subregion++;
    }

    auto region_ext = this->make_region_ext(si);

    // Create a union of all outer nodes
    NodeId result = vb.insert_region(Label{label_, region_ext + ".ou"},
                                     Joined{op_or, std::move(outer_nodes)});

    if (!inner_nodes.empty())
    {
        // Create a union of all inner nodes
        NodeId inner_union
            = vb.insert_region(Label{label_, region_ext + ".iu"},
                               Joined{op_or, std::move(inner_nodes)});

        // Create a negation of this union
        auto negation = vb.insert_region(Label{label_, region_ext + ".nui"},
                                         Negated{inner_union});

        // Subtract the inner union from the outer union
        result = vb.insert_region(Label{label_, region_ext + ".d"},
                                  Joined{op_and, {result, negation}});
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Make a translated cylinder node.
 */
NodeId RevolvedPolygon::make_cylinder(detail::VolumeBuilder& vb,
                                      Real2 const& p0,
                                      Real2 const& p1,
                                      SubIndex const& si) const
{
    real_type hh = std::fabs(p1[Z] - p0[Z]) / 2;
    auto z_bot = std::min(p0[Z], p1[Z]);
    auto scoped_transform
        = vb.make_scoped_transform(Translation({0, 0, hh + z_bot}));
    Cylinder local_cyl{p0[R], hh};
    return build_intersect_region(
        vb, std::string{label_}, this->make_subregion_ext(si), local_cyl);
}

//---------------------------------------------------------------------------//
/*!
 * Make a translated cone node.
 */
NodeId RevolvedPolygon::make_cone(detail::VolumeBuilder& vb,
                                  Real2 const& p0,
                                  Real2 const& p1,
                                  SubIndex const& si) const
{
    Real2 p_bot = p0;
    Real2 p_top = p1;

    if (p_bot[Z] > p_top[Z])
    {
        std::swap(p_bot, p_top);
    }

    real_type hh = 0.5 * (p_top[Z] - p_bot[Z]);
    Real2 radii{p_bot[R], p_top[R]};

    auto scoped_transform
        = vb.make_scoped_transform(Translation({0, 0, hh + p_bot[Z]}));
    Cone local_cone{radii, hh};
    return build_intersect_region(
        vb, std::string{label_}, this->make_subregion_ext(si), local_cone);
}

//---------------------------------------------------------------------------//
/*!
 * Make a label extension for a level.
 */
std::string RevolvedPolygon::make_level_ext(RevolvedPolygon::SubIndex si) const
{
    return std::to_string(si.level);
}

//---------------------------------------------------------------------------//
/*!
 * Make a label extension for a region within a level.
 */
std::string RevolvedPolygon::make_region_ext(RevolvedPolygon::SubIndex si) const
{
    return this->make_level_ext(si) + "." + std::to_string(si.region);
}

//---------------------------------------------------------------------------//
/*!
 * Make a label extension for a subregion within a region.
 */
std::string
RevolvedPolygon::make_subregion_ext(RevolvedPolygon::SubIndex si) const
{
    return this->make_region_ext(si) + "." + std::to_string(si.subregion);
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
