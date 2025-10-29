//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/StackedExtrudedPolygon.cc
//---------------------------------------------------------------------------//
#include "StackedExtrudedPolygon.hh"

#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"

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
    X = 0,
    Y = 1,
    Z = 2
};
}  // namespace
//---------------------------------------------------------------------------//
/*!
 * Construct, or return an ExtrudedPolygon if possible.
 */
auto StackedExtrudedPolygon::or_solid(std::string&& label,
                                      VecReal2&& polygon,
                                      VecReal3&& polyline,
                                      VecReal&& scaling) -> SPConstObject
{
    CELER_VALIDATE(polygon.size() >= 3,
                   << "polygon must have at least 3 vertices");
    CELER_VALIDATE(polyline.size() >= 2,
                   << "polyline must have at least 2 vertices");
    CELER_VALIDATE(polyline.size() == scaling.size(),
                   << "polyline and scaling must be the same size");

    // If the polygon is convex and the polyline is a single segment, make an
    // ExtrudedPolygon
    if (polyline.size() == 2
        && detail::is_convex(make_span(polygon), /* degen_ok = */ true))
    {
        constexpr auto bot = to_int(Bound::lo);
        constexpr auto top = to_int(Bound::hi);

        CELER_VALIDATE(polyline[bot][Z] < polyline[top][Z],
                       << "z coordinates must be strictly increasing");
        CELER_VALIDATE(scaling[bot] > 0 && scaling[top] > 0,
                       << "scaling values must be positive");

        // Create a single ExtrudedPolygon shape
        ExtrudedPolygon ep(polygon,
                           {polyline[bot], scaling[bot]},
                           {polyline[top], scaling[top]});
        return std::make_shared<ExtrudedPolygonShape>(std::move(label),
                                                      std::move(ep));
    }

    // Concave polygon or multiple segments: create a StackedExtrudedPolygon
    return std::make_shared<StackedExtrudedPolygon>(std::move(label),
                                                    std::move(polygon),
                                                    std::move(polyline),
                                                    std::move(scaling));
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a polygon, polyline, and scaling factors.
 */
StackedExtrudedPolygon::StackedExtrudedPolygon(std::string&& label,
                                               VecReal2&& polygon,
                                               VecReal3&& polyline,
                                               VecReal&& scaling)
    : label_{std::move(label)}
    , polygon_{std::move(polygon)}
    , polyline_{std::move(polyline)}
    , scaling_{std::move(scaling)}
{
    CELER_VALIDATE(polygon_.size() >= 3,
                   << "polygon must have at least 3 vertices");
    CELER_VALIDATE(polyline_.size() >= 2,
                   << "polyline must have at least 2 vertices");
    CELER_VALIDATE(polyline_.size() == scaling_.size(),
                   << "polyline and scaling must be the same size");

    // Validate that z coordinates are nondecreasing
    CELER_VALIDATE(std::adjacent_find(polyline_.begin(),
                                      polyline_.end(),
                                      [](auto const& a, auto const& b) {
                                          return a[Z] > b[Z];
                                      })
                       == polyline_.end(),
                   << "z coordinates must be nondecreasing");

    // Validate scaling factors
    CELER_VALIDATE(std::all_of(scaling_.begin(),
                               scaling_.end(),
                               [](auto& s) { return s >= 0; }),
                   << "scaling factor must be nonnegative");
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume from this shape.
 */
NodeId StackedExtrudedPolygon::build(VolumeBuilder& vb) const
{
    // Recursively handle convex decomposition
    SubRegionIndex si;
    return this->make_levels(vb, polygon_, si);
}

//---------------------------------------------------------------------------//
/*!
 * Write the shape to JSON.
 */
void StackedExtrudedPolygon::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}
//---------------------------------------------------------------------------//
// HELPER FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Recursively construct stacks, subtracting out concavities.
 */
NodeId StackedExtrudedPolygon::make_levels(
    detail::VolumeBuilder& vb,
    VecReal2 const& polygon,
    StackedExtrudedPolygon::SubRegionIndex si) const
{
    // Find the convex hull
    detail::ConvexHullFinder<real_type> hull_finder(polygon, vb.tol());
    auto convex_hull = hull_finder.make_convex_hull();
    auto concave_regions = hull_finder.calc_concave_regions();

    // Build the convex hull stack
    NodeId result = this->make_stack(vb, convex_hull, si);

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
            vb, concave_regions[i], SubRegionIndex{si.level + 1, i});
    }

    auto level_label = this->make_level_ext(si);

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
 * Extrude a *convex* polygon along the polyline.
 */
NodeId
StackedExtrudedPolygon::make_stack(detail::VolumeBuilder& vb,
                                   VecReal2 const& polygon,
                                   StackedExtrudedPolygon::SubRegionIndex si) const
{
    std::vector<NodeId> nodes;
    SoftEqual<real_type> soft_equal(vb.tol().rel, vb.tol().abs);
    SoftZero<real_type> soft_zero(vb.tol().abs);

    // Add to the stack: all polyline segments with non-zero z length and
    // non-zero radii
    for (auto i : range(polyline_.size() - 1))
    {
        CELER_VALIDATE(soft_zero(scaling_[i]) == soft_zero(scaling_[i + 1])
                           || soft_equal(polyline_[i][Z], polyline_[i + 1][Z]),
                       << "non-zero-length polyline segment cannot have "
                          "scaling = 0 on exactly one z plane");

        if (!soft_equal(polyline_[i][Z], polyline_[i + 1][Z])
            && !soft_zero(scaling_[i]))
        {
            // Create the ExtrudedPolygon for this segment
            ExtrudedPolygon shape{polygon,
                                  {polyline_[i], scaling_[i]},
                                  {polyline_[i + 1], scaling_[i + 1]}};

            // Build this segment with unique label
            nodes.push_back(build_intersect_region(
                vb, label_, this->make_segment_ext(si, i), shape));
        }
    }

    // Create a union of all segments
    return vb.insert_region(Label{label_, this->make_stack_ext(si)},
                            Joined{op_or, std::move(nodes)});
}

//---------------------------------------------------------------------------//
/*!
 * Make a label extension for a level.
 */
std::string StackedExtrudedPolygon::make_level_ext(
    StackedExtrudedPolygon::SubRegionIndex si) const
{
    return std::to_string(si.level);
}

//---------------------------------------------------------------------------//
/*!
 * Make a label extension for a stack within a level.
 */
std::string StackedExtrudedPolygon::make_stack_ext(
    StackedExtrudedPolygon::SubRegionIndex si) const
{
    return this->make_level_ext(si) + "." + std::to_string(si.stack);
}

//---------------------------------------------------------------------------//
/*!
 * Make a label extension for a segment within a stack.
 */
std::string StackedExtrudedPolygon::make_segment_ext(
    StackedExtrudedPolygon::SubRegionIndex si, size_type segment_idx) const
{
    return this->make_stack_ext(si) + "." + std::to_string(segment_idx);
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
