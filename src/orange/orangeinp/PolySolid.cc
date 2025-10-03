//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/PolySolid.cc
//---------------------------------------------------------------------------//
#include "PolySolid.hh"

#include "corecel/cont/Range.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/transform/Translation.hh"

#include "ObjectIO.json.hh"
#include "Transformed.hh"

#include "detail/BuildIntersectRegion.hh"
#include "detail/VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
namespace
{
//---------------------------------------------------------------------------//
//! Construct the unioned "interior" of a polysolid
template<class T>
[[nodiscard]] NodeId construct_segments(PolySolidBase const& base,
                                        T&& build_region,
                                        detail::VolumeBuilder& vb)
{
    std::string const label{base.label()};
    auto const& segments = base.segments();
    CELER_ASSERT(segments.z().size() == segments.size() + 1);

    SoftEqual soft_eq{vb.tol().rel};
    std::vector<NodeId> segment_nodes;

    for (auto i : range(segments.size()))
    {
        // Translate this segment along z
        auto const [zlo, zhi] = segments.z(i);
        if (soft_eq(zlo, zhi))
        {
            // Effectively zero height segment (degenerate: perhaps stacked
            // cylinders, for example)
            continue;
        }
        real_type const hz = (zhi - zlo) / 2;
        auto scoped_transform
            = vb.make_scoped_transform(Translation{{0, 0, zlo + hz}});

        // Build outer shape
        NodeId segment_node;
        {
            auto outer = build_region(segments.outer(i), hz);
            segment_node = build_intersect_region(
                vb, std::string{label}, std::to_string(i) + ".interior", outer);
        }

        if (segments.has_exclusion())
        {
            // Build inner shape
            auto inner = build_region(segments.inner(i), hz);
            NodeId inner_node = build_intersect_region(
                vb, std::string{label}, std::to_string(i) + ".excluded", inner);

            // Subtract (i.e., "and not") inner shape from this segment
            auto sub_node = vb.insert_region({}, Negated{inner_node});
            segment_node
                = vb.insert_region(Label{label, std::to_string(i)},
                                   Joined{op_and, {segment_node, sub_node}});
        }
        segment_nodes.push_back(segment_node);
    }

    // Union the given segments to create a new CSG node
    return vb.insert_region(Label{label, "segments"},
                            Joined{op_or, std::move(segment_nodes)});
}

//---------------------------------------------------------------------------//
/*!
 * Construct an enclosed angle if applicable.
 */
[[nodiscard]] NodeId construct_enclosed_angle(PolySolidBase const& base,
                                              detail::VolumeBuilder& vb,
                                              NodeId result)
{
    if (auto const& azi = base.enclosed_azi())
    {
        // The enclosed angle is "true" (specified by the user to truncate the
        // shape azimuthally): construct a wedge to be added or deleted
        auto&& [sense, wedge] = azi.make_sense_region();
        char const* ext = (sense == Sense::outside ? "~azi" : "azi");
        NodeId wedge_id = build_intersect_region(vb, base.label(), ext, wedge);
        if (sense == Sense::outside)
        {
            wedge_id = vb.insert_region({}, Negated{wedge_id});
        }
        result
            = vb.insert_region(Label{std::string{base.label()}, "restricted"},
                               Joined{op_and, {result, wedge_id}});
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a filled polygon solid.
 */
PolySegments::PolySegments(VecReal&& outer, VecReal&& z)
    : PolySegments{{}, std::move(outer), std::move(z)}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a shell of a polygon solid.
 */
PolySegments::PolySegments(VecReal&& inner, VecReal&& outer, VecReal&& z)
    : inner_{std::move(inner)}, outer_{std::move(outer)}, z_{std::move(z)}
{
    CELER_VALIDATE(z_.size() >= 2,
                   << "no axial segments was specified: at least 2 points "
                      "needed (given "
                   << z_.size() << ")");
    CELER_VALIDATE(outer_.size() == z_.size(),
                   << "inconsistent outer radius size (" << outer_.size()
                   << "): expected " << z_.size());
    CELER_VALIDATE(inner_.empty() || inner_.size() == z_.size(),
                   << "inconsistent inner radius size (" << inner_.size()
                   << "): expected " << z_.size());

    if (z_.front() > z_.back())
    {
        // Input grid is decreasing: reverse everything
        std::reverse(z_.begin(), z_.end());
        std::reverse(inner_.begin(), inner_.end());
        std::reverse(outer_.begin(), outer_.end());
    }
    CELER_VALIDATE(is_monotonic_nondecreasing(make_span(z_)),
                   << "axial grid is non-monotonic");
    for (auto i : range(outer_.size()))
    {
        CELER_VALIDATE(outer_[i] >= 0, << "invalid outer radius " << outer_[i]);
        CELER_VALIDATE(inner_.empty()
                           || (inner_[i] >= 0 && inner_[i] <= outer_[i]),
                       << "invalid inner radius " << inner_[i]);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Build with label, axial segments, optional restriction.
 */
PolySolidBase::PolySolidBase(std::string&& label,
                             PolySegments&& segments,
                             EnclosedAzi&& enclosed)
    : label_{std::move(label)}
    , segments_{std::move(segments)}
    , enclosed_{std::move(enclosed)}
{
}

//---------------------------------------------------------------------------//
//! Anchored default virtual destructor
PolySolidBase::~PolySolidBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Return a polycone *or* a simplified version for only a single segment.
 */
auto PolyCone::or_solid(std::string&& label,
                        PolySegments&& segments,
                        EnclosedAzi&& enclosed) -> SPConstObject
{
    if (segments.size() > 1)
    {
        // Can't be simplified: make a polycone
        return std::make_shared<PolyCone>(
            std::move(label), std::move(segments), std::move(enclosed));
    }

    auto const [zlo, zhi] = segments.z(0);
    real_type const hh = (zhi - zlo) / 2;

    Cone outer{segments.outer(0), hh};
    std::optional<Cone> inner;
    if (segments.has_exclusion())
    {
        inner = Cone{segments.inner(0), hh};
    }

    auto result = ConeSolid::or_shape(std::move(label),
                                      std::move(outer),
                                      std::move(inner),
                                      std::move(enclosed));
    if (real_type dz = (zhi + zlo) / 2; dz != 0)
    {
        result = std::make_shared<Transformed>(std::move(result),
                                               Translation{{0, 0, dz}});
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Build with label, axial segments, optional restriction.
 */
PolyCone::PolyCone(std::string&& label,
                   PolySegments&& segments,
                   EnclosedAzi&& enclosed)
    : PolySolidBase{std::move(label), std::move(segments), std::move(enclosed)}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume from this shape.
 */
NodeId PolyCone::build(VolumeBuilder& vb) const
{
    auto build_cone
        = [](Real2 const& radii, real_type hh) { return Cone{radii, hh}; };

    // Construct union of all cone segments
    NodeId result = construct_segments(*this, build_cone, vb);

    /*!
     * \todo After adding short-circuit logic to evaluator, add "acceleration"
     * structures here, e.g. "inside(inner cylinder) || [inside(outer cylinder)
     * && (original union)]"
     */

    // Construct azimuthal truncation if applicable
    result = construct_enclosed_angle(*this, vb, result);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Write the shape to JSON.
 */
void PolyCone::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
/*!
 * Return a polycone *or* a simplified version for only a single segment.
 */
auto PolyPrism::or_solid(std::string&& label,
                         PolySegments&& segments,
                         EnclosedAzi&& enclosed,
                         int num_sides,
                         real_type orientation) -> SPConstObject
{
    if (segments.size() > 1)
    {
        // Can't be simplified: make a polycone
        return std::make_shared<PolyPrism>(std::move(label),
                                           std::move(segments),
                                           std::move(enclosed),
                                           num_sides,
                                           orientation);
    }

    auto const [zlo, zhi] = segments.z(0);
    real_type const hh = (zhi - zlo) / 2;

    auto const [rlo, rhi] = segments.outer(0);
    if (rlo != rhi)
    {
        CELER_NOT_IMPLEMENTED("prism with different lo/hi radii");
    }

    Prism outer{num_sides, rlo, hh, orientation};
    std::optional<Prism> inner;
    if (segments.has_exclusion())
    {
        auto const [rilo, rihi] = segments.inner(0);
        if (rilo != rihi)
        {
            CELER_NOT_IMPLEMENTED("prism with different lo/hi radii");
        }

        inner = Prism{num_sides, rilo, hh, orientation};
    }

    auto result = PrismSolid::or_shape(std::move(label),
                                       std::move(outer),
                                       std::move(inner),
                                       std::move(enclosed));
    if (real_type dz = (zhi + zlo) / 2; dz != 0)
    {
        result = std::make_shared<Transformed>(std::move(result),
                                               Translation{{0, 0, dz}});
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Build with label, axial segments, optional restriction.
 */
PolyPrism::PolyPrism(std::string&& label,
                     PolySegments&& segments,
                     EnclosedAzi&& enclosed,
                     int num_sides,
                     real_type orientation)
    : PolySolidBase{std::move(label), std::move(segments), std::move(enclosed)}
    , num_sides_{num_sides}
    , orientation_{orientation}
{
    CELER_VALIDATE(num_sides_ >= 3,
                   << "degenerate prism (num_sides = " << num_sides_ << ')');
    CELER_VALIDATE(orientation_ >= 0 && orientation_ < 1,
                   << "orientation is out of bounds [0, 1): " << orientation_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume from this shape.
 */
NodeId PolyPrism::build(VolumeBuilder& vb) const
{
    auto build_prism = [this](Real2 const& radii, real_type hh) {
        if (radii[0] != radii[1])
        {
            CELER_NOT_IMPLEMENTED("prism with different lo/hi radii");
        }
        return Prism{this->num_sides_, radii[0], hh, this->orientation_};
    };

    // Construct union of all cone segments
    NodeId result = construct_segments(*this, build_prism, vb);

    /*!
     *\todo After adding short-circuit logic to evaluator, add "acceleration"
     * structures here, e.g. "inside(inner cylinder) || [inside(outer cylinder)
     * && (original union)]"
     */

    // Construct azimuthal truncation if applicable
    result = construct_enclosed_angle(*this, vb, result);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Write the shape to JSON.
 */
void PolyPrism::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
