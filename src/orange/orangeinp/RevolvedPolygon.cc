//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/RevolvedPolygon.cc
//---------------------------------------------------------------------------//
#include "RevolvedPolygon.hh"

#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "orange/transform/Translation.hh"

#include "ObjectIO.json.hh"
#include "Shape.hh"
#include "Transformed.hh"

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
 * Construct from a polygon, polyline, and scaling factors.
 */
RevolvedPolygon::RevolvedPolygon(std::string&& label, VecReal2&& polygon)
    : label_{std::move(label)}, polygon_{std::move(polygon)}
{
    CELER_VALIDATE(polygon_.size() >= 3,
                   << "polygon must have at least 3 vertices");
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume from this shape.
 */
NodeId RevolvedPolygon::build(VolumeBuilder& vb) const
{
    //// Use the volume builder's tolerance to remove any colinear points
    // polygon_ = detail::filter_collinear_points(polygon_, vb.tol());

    //// After removing collinear points, at least 3 points must remain
    // CELER_VALIDATE(polygon_.size() >= 3,
    //                << "polygon must consist of at least 3 points");

    //// After removing collinear points, the polygon should have a *strictly*
    //// clockwise orientation, which also guarantees it is convex.
    // CELER_VALIDATE(
    //     has_orientation(make_span(polygon_),
    //     detail::Orientation::clockwise),
    //     << "polygon must be specified in strictly clockwise order");

    //// All points must be positive
    // CELER_VALIDATE(
    //     std::all_of(polygon_.begin(),
    //                 polygon_.end(),
    //                 [](Real2 const& p) { return p[R] >= 0 && p[Z] >= 0; }),
    //     << "polygon must consist of only positive r and z values");

    // Start the recursion process

    SubregionIndex ri;
    return this->make_levels(vb, polygon_, ri);
}

//---------------------------------------------------------------------------//
/*!
 * Write the shape to JSON.
 */
void RevolvedPolygon::output(JsonPimpl* j) const
{
    // to_json_pimpl(j, *this);
}
//---------------------------------------------------------------------------//
// HELPER FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Recursively construct convex revolved polygons, subtracting out concavities.
 */
NodeId RevolvedPolygon::make_levels(detail::VolumeBuilder& vb,
                                    VecReal2 const& polygon,
                                    RevolvedPolygon::SubregionIndex si) const
{
    // Find the convex hull
    detail::ConvexHullFinder<real_type> hull_finder(polygon, vb.tol());
    auto convex_hull = hull_finder.make_convex_hull();
    auto concave_regions = hull_finder.calc_concave_regions();

    // Build the convex region
    NodeId result = this->make_region(vb, convex_hull, si);

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
            vb, concave_regions[i], SubregionIndex{si.level + 1, i});
    }

    auto level_label = this->make_level_label(si);

    // Create a union of all concave regions
    NodeId concave_union
        = vb.insert_region(Label{label_, level_label + ".cu"},
                           Joined{op_or, std::move(concave_nodes)});

    // Create a negation of this union
    auto sub_node = vb.insert_region(Label{label_, level_label + ".ncu"},
                                     Negated{concave_union});

    // Subtract concave regions from the convex hull
    return vb.insert_region(Label{label_, level_label + ".d"},
                            Joined{op_and, {result, sub_node}});
}

//---------------------------------------------------------------------------//
/*!
 * Revolved a *convex* polygon around the \em z axis.
 */
NodeId RevolvedPolygon::make_region(detail::VolumeBuilder& vb,
                                    VecReal2 const& polygon,
                                    SubregionIndex si) const
{
    SoftEqual<real_type> soft_equal(vb.tol().rel, vb.tol().abs);
    SoftZero<real_type> soft_zero(vb.tol().abs);

    auto n = polygon.size();
    std::vector<NodeId> outer_nodes;
    std::vector<NodeId> inner_nodes;

    auto next_idx = [n](size_type i) { return (i + 1 < n) ? i + 1 : 0; };

    for (auto i : range(n))
    {
        auto const& p0 = polygon[i];
        auto const& p1 = polygon[next_idx(i)];

        if (soft_equal(p0[Z], p1[Z]) || (soft_zero(p0[R]) && soft_zero(p1[R])))
        {
            // Perpendicular to or coincide with z, don't make a shape
            continue;
        }

        // Make a cylinder or  cone, and add it to the inner/outer nodes vector
        NodeId shape_id = soft_equal(p0[R], p1[R])
                              ? this->make_cylinder(vb, p0, p1, si)
                              : this->make_cone(vb, p0, p1, si);
        (p0[Z] < p1[Z] ? outer_nodes : inner_nodes).push_back(shape_id);

        si.subregion++;
    }

    auto region_label = this->make_region_label(si);

    // Create a union of all outer nodes
    NodeId result = vb.insert_region(Label{label_, region_label + ".ou"},
                                     Joined{op_or, std::move(outer_nodes)});

    if (!inner_nodes.empty())
    {
        // Create a union of all inner nodes
        NodeId inner_union
            = vb.insert_region(Label{label_, region_label + ".iu"},
                               Joined{op_or, std::move(inner_nodes)});

        // Create a negation of this union
        auto negation = vb.insert_region(Label{label_, region_label + ".niu"},
                                         Negated{inner_union});

        // Subtract concave regions from the convex hull
        result = vb.insert_region(Label{label_, region_label + ".d"},
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
                                      SubregionIndex const& si) const
{
    auto subregion_label = this->make_subregion_label(si);

    real_type hh = std::abs(p1[Z] - p0[Z]) / 2;
    auto cyl = std::make_shared<CylinderShape>(
        std::move(subregion_label), p0[R], hh);

    auto bot_z = std::min(p0[Z], p1[Z]);
    Transformed trans(cyl, Translation({0, 0, bot_z + hh}));
    return trans.build(vb);
}

//---------------------------------------------------------------------------//
/*!
 * Make a translated cone node
 */
NodeId RevolvedPolygon::make_cone(detail::VolumeBuilder& vb,
                                  Real2 const& p0,
                                  Real2 const& p1,
                                  SubregionIndex const& si) const
{
    auto [r_bot, r_top] = (p0[Z] < p1[Z]) ? std::pair{p0[R], p1[R]}
                                          : std::pair{p1[R], p0[R]};
    auto [z_bot, z_top] = std::minmax(p0[Z], p1[Z]);

    real_type hh = 0.5 * (z_top - z_bot);

    Real2 radii{r_bot, r_top};
    auto subregion_label = this->make_subregion_label(si);
    auto cone
        = std::make_shared<ConeShape>(std::move(subregion_label), radii, hh);

    return Transformed(cone, Translation({0, 0, hh + z_bot})).build(vb);
}

//---------------------------------------------------------------------------//
/*!
 * Make a label for a level.
 */
std::string
RevolvedPolygon::make_level_label(RevolvedPolygon::SubregionIndex ri) const
{
    return std::to_string(ri.level);
}

//---------------------------------------------------------------------------//
/*!
 * Make a label for a region within a level.
 */
std::string
RevolvedPolygon::make_region_label(RevolvedPolygon::SubregionIndex ri) const
{
    return this->make_level_label(ri) + "." + std::to_string(ri.region);
}

//---------------------------------------------------------------------------//
/*!
 * Make a label for a subregion within a region.
 */
std::string
RevolvedPolygon::make_subregion_label(RevolvedPolygon::SubregionIndex ri) const
{
    return this->make_level_label(ri) + "." + std::to_string(ri.region) + "."
           + std::to_string(ri.subregion);
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
