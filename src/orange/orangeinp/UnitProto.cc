//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/UnitProto.cc
//---------------------------------------------------------------------------//
#include "UnitProto.hh"

#include <algorithm>
#include <numeric>

#include "orange/OrangeData.hh"
#include "orange/OrangeInput.hh"

#include "CsgObject.hh"
#include "CsgTree.hh"
#include "CsgTreeUtils.hh"
#include "Transformed.hh"

#include "detail/CsgUnit.hh"
#include "detail/CsgUnitBuilder.hh"
#include "detail/InputBuilder.hh"
#include "detail/InternalSurfaceFlagger.hh"
#include "detail/PostfixLogicBuilder.hh"
#include "detail/VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Construct with required input data.
 */
UnitProto::UnitProto(Input&& inp) : input_{std::move(inp)}
{
    CELER_VALIDATE(input_, << "no fill, daughters, or volumes are defined");
    CELER_VALIDATE(std::all_of(input_.materials.begin(),
                               input_.materials.begin(),
                               [](MaterialInput const& m) {
                                   return static_cast<bool>(m);
                               }),
                   << "incomplete material definition(s)");
    CELER_VALIDATE(std::all_of(input_.daughters.begin(),
                               input_.daughters.begin(),
                               [](DaughterInput const& d) {
                                   return static_cast<bool>(d);
                               }),
                   << "incomplete daughter definition(s)");
    CELER_VALIDATE(input_.boundary.zorder == ZOrder::media
                       || input_.boundary.zorder == ZOrder::exterior,
                   << "invalid exterior zorder '"
                   << to_cstring(input_.boundary.zorder) << "'");
}

//---------------------------------------------------------------------------//
/*!
 * Short unique name of this object.
 */
std::string_view UnitProto::label() const
{
    return input_.label;
}

//---------------------------------------------------------------------------//
/*!
 * Get the boundary of this universe as an object.
 */
auto UnitProto::interior() const -> SPConstObject
{
    return input_.boundary.interior;
}

//---------------------------------------------------------------------------//
/*!
 * Get a list of all daughter protos.
 */
auto UnitProto::daughters() const -> VecProto
{
    VecProto result;
    for (auto const& d : input_.daughters)
    {
        result.push_back(d.fill.get());
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a universe input from this object.
 *
 * Construction is done from highest masking precedence to lowest (reverse
 * zorder): exterior, then holes, then arrays, then media.
 */
void UnitProto::build(InputBuilder& input) const
{
    // Build CSG unit
    auto csg_unit = this->build(input.tol(),
                                input.next_id() == orange_global_universe
                                    ? ExteriorBoundary::is_global
                                    : ExteriorBoundary::is_daughter);
    CELER_ASSERT(csg_unit);

    // Get the list of all surfaces actually used
    auto const sorted_local_surfaces = calc_surfaces(csg_unit.tree);
    bool const has_background = csg_unit.background != MaterialId{};

    UnitInput result;
    result.label = input_.label;

    // Save unit's bounding box (inverted bounding zone of exterior)
    {
        NodeId node_id = csg_unit.volumes[orange_exterior_volume.get()];
        auto region_iter = csg_unit.regions.find(node_id);
        CELER_ASSERT(region_iter != csg_unit.regions.end());
        auto const& bz = region_iter->second.bounds;
        if (bz.negated)
        {
            result.bbox = bz.interior;
        }
    }

    // Save surfaces
    result.surfaces.reserve(sorted_local_surfaces.size());
    for (auto const& lsid : sorted_local_surfaces)
    {
        result.surfaces.emplace_back(csg_unit.surfaces[lsid.get()]);
    }

    // Save surface labels
    result.surface_labels.resize(result.surfaces.size());
    for (auto node_id : range(NodeId{csg_unit.tree.size()}))
    {
        if (auto* surf_node = std::get_if<Surface>(&csg_unit.tree[node_id]))
        {
            LocalSurfaceId old_lsid = surf_node->id;
            auto idx = static_cast<size_type>(
                find_sorted(sorted_local_surfaces.begin(),
                            sorted_local_surfaces.end(),
                            old_lsid)
                - sorted_local_surfaces.begin());
            CELER_ASSERT(idx < result.surface_labels.size());

            // NOTE: surfaces may be created more than once. Our primitive
            // "input" allows association with only one surface, so we'll
            // arbitrarily choose the lexicographically sorted "first" surface
            // name in the list.
            CELER_ASSERT(!csg_unit.metadata[node_id.get()].empty());
            auto const& label = *csg_unit.metadata[node_id.get()].begin();
            result.surface_labels[idx] = label;
        }
    }

    // Loop over all volumes to construct
    detail::PostfixLogicBuilder build_logic{csg_unit.tree,
                                            sorted_local_surfaces};
    detail::InternalSurfaceFlagger has_internal_surfaces{csg_unit.tree};
    result.volumes.reserve(csg_unit.volumes.size() + has_background);

    for (auto vol_idx : range(csg_unit.volumes.size()))
    {
        NodeId node_id = csg_unit.volumes[vol_idx];
        VolumeInput vi;

        // Construct logic and faces with remapped surfaces
        auto&& [faces, logic] = build_logic(node_id);
        vi.faces = std::move(faces);
        vi.logic = std::move(logic);

        // Set bounding box
        auto region_iter = csg_unit.regions.find(node_id);
        CELER_ASSERT(region_iter != csg_unit.regions.end());
        vi.bbox = get_exterior_bbox(region_iter->second.bounds);
        /* TODO: "simple safety" flag is set inside
         * "unit inserter" (move here once we stop importing from SCALE via
         * OrangeInput)
         */
        if (has_internal_surfaces(node_id))
        {
            vi.flags |= VolumeRecord::internal_surfaces;
        }

        vi.zorder = ZOrder::media;
        result.volumes.emplace_back(std::move(vi));
    }

    if (has_background)
    {
        // "Background" should be unreachable: 'nowhere' logic, null bbox
        // but it has to have all the surfaces that connect to an interior
        // volume
        VolumeInput vi;
        vi.faces.resize(sorted_local_surfaces.size());
        std::iota(vi.faces.begin(), vi.faces.end(), LocalSurfaceId{0});
        vi.logic = {logic::ltrue, logic::lnot};
        vi.bbox = {};  // XXX: input converter changes to infinite bbox
        vi.zorder = ZOrder::background;
        vi.flags = VolumeRecord::implicit_vol;
        result.volumes.emplace_back(std::move(vi));
    }
    CELER_ASSERT(result.volumes.size()
                 == csg_unit.volumes.size() + has_background);

    // Set labels and other attributes.
    // NOTE: this means we're entirely ignoring the "metadata" from the CSG
    // nodes for the region, because we can't know which ones have the
    // user-supplied volume names
    // TODO: add JSON output to the input builder that includes the CSG
    // metadata
    auto vol_iter = result.volumes.begin();

    // Save attributes for exterior volume
    if (input.next_id() != orange_global_universe)
    {
        vol_iter->zorder = ZOrder::implicit_exterior;
        vol_iter->flags |= VolumeRecord::implicit_vol;
    }
    else
    {
        vol_iter->zorder = input_.boundary.zorder;
    }
    vol_iter->label = {"[EXTERIOR]", input_.label};
    ++vol_iter;

    for (auto const& d : input_.daughters)
    {
        LocalVolumeId const vol_id{
            static_cast<size_type>(vol_iter - result.volumes.begin())};

        // Save daughter volume attributes
        vol_iter->label
            = Label{std::string{d.fill->label()}, std::string{this->label()}};
        vol_iter->zorder = d.zorder;
        /* TODO: the "embedded_universe" flag is *also* set by the unit
         * builder. Move that here. */
        ++vol_iter;

        // Add daughter to map
        auto&& [iter, inserted] = result.daughter_map.insert({vol_id, {}});
        CELER_ASSERT(inserted);
        // Convert proto pointer to universe ID
        iter->second.universe_id = input.find_universe_id(d.fill.get());

        // Save the transform
        auto const* fill = std::get_if<Daughter>(&csg_unit.fills[vol_id.get()]);
        CELER_ASSERT(fill);
        auto transform_id = fill->transform_id;
        CELER_ASSERT(transform_id < csg_unit.transforms.size());
        iter->second.transform = csg_unit.transforms[transform_id.get()];
    }

    // Save attributes from materials
    for (auto const& m : input_.materials)
    {
        vol_iter->label = std::string{m.interior->label()};
        vol_iter->zorder = ZOrder::media;
        ++vol_iter;
    }

    if (input_.fill)
    {
        vol_iter->label = {input_.label, "bg"};
        ++vol_iter;
    }
    CELER_EXPECT(vol_iter == result.volumes.end());

    // TODO: save material IDs as well
    input.insert(std::move(result));
}

//---------------------------------------------------------------------------//
/*!
 * Construct a standalone unit for testing or external interface.
 *
 * The universe ID in the resulting "fill" is the index into the input \c
 * daughters list, which is the same ordering as the array of \c
 * this->daughters().
 *
 * The "exterior boundary" argument determines whether the outer boundary needs
 * to be deleted (assumed inside, implicit from the parent universe's boundary)
 * or preserved.
 */
auto UnitProto::build(Tol const& tol, ExteriorBoundary ext) const -> Unit
{
    CELER_EXPECT(tol);
    detail::CsgUnit result;
    detail::CsgUnitBuilder unit_builder(&result, tol);

    auto build_volume = [ub = &unit_builder](ObjectInterface const& obj) {
        detail::VolumeBuilder vb{ub};
        auto final_node = obj.build(vb);
        return ub->insert_volume(final_node);
    };

    // Build exterior volume and optional background fill
    if (input_.boundary.zorder != ZOrder::media && !input_.fill)
    {
        CELER_NOT_IMPLEMENTED("implicit exterior without background fill");
    }
    auto ext_vol
        = build_volume(NegatedObject("[EXTERIOR]", input_.boundary.interior));
    CELER_ASSERT(ext_vol == orange_exterior_volume);

    // Build daughters
    UniverseId daughter_id{0};
    for (auto const& d : input_.daughters)
    {
        if (d.zorder != ZOrder::media)
        {
            CELER_NOT_IMPLEMENTED("volume masking using different z orders");
        }
        auto lv = build_volume(*d.make_interior());
        unit_builder.fill_volume(lv, daughter_id++, d.transform);
    }

    // Build materials
    for (auto const& m : input_.materials)
    {
        auto lv = build_volume(*m.interior);
        unit_builder.fill_volume(lv, m.fill);
    }

    // Build background fill (optional)
    result.background = input_.fill;

    if (ext == ExteriorBoundary::is_daughter)
    {
        // Replace "exterior" with "False" (i.e. interior with true)
        NodeId ext_node = result.volumes[ext_vol.unchecked_get()];
        auto min_node = replace_down(&result.tree, ext_node, False{});
        // Simplify recursively
        simplify(&result.tree, min_node);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct the daughter's shape in this unit's reference frame.
 */
auto UnitProto::DaughterInput::make_interior() const -> SPConstObject
{
    CELER_EXPECT(*this);
    return Transformed::or_object(this->fill->interior(),
                                  VariantTransform{this->transform});
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
