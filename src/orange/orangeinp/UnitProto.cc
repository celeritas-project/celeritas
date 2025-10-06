//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/UnitProto.cc
//---------------------------------------------------------------------------//
#include "UnitProto.hh"

#include <algorithm>
#include <numeric>
#include <utility>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/OpaqueIdIO.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/LabelIO.json.hh"
#include "corecel/io/Logger.hh"
#include "geocel/VolumeToString.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeInput.hh"
#include "orange/transform/VariantTransform.hh"

#include "CsgObject.hh"
#include "CsgTree.hh"
#include "CsgTreeIO.json.hh"
#include "CsgTreeUtils.hh"
#include "ObjectIO.json.hh"
#include "Transformed.hh"

#include "detail/CsgLogicUtils.hh"
#include "detail/CsgUnit.hh"
#include "detail/CsgUnitBuilder.hh"
#include "detail/InternalSurfaceFlagger.hh"
#include "detail/ProtoBuilder.hh"
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
                               Identity{}),
                   << "incomplete material definition(s)");
    CELER_VALIDATE(std::all_of(input_.daughters.begin(),
                               input_.daughters.begin(),
                               Identity{}),
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
    return input_.label.name;
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
void UnitProto::build(ProtoBuilder& input) const
{
    // Bounding box should be finite if and only if this is the global universe
    CELER_EXPECT((input.next_id() == orange_global_universe)
                 == !input.bbox(input.next_id()));

    // Build CSG unit
    auto csg_unit = this->build(input.tol(), input.bbox(input.next_id()));
    CELER_ASSERT(csg_unit);

    // Get the list of all surfaces actually used
    auto const sorted_local_surfaces = calc_surfaces(csg_unit.tree);
    CELER_LOG(debug) << "...built " << this->label() << ": used "
                     << sorted_local_surfaces.size() << " of "
                     << csg_unit.surfaces.size() << " surfaces";

    UnitInput result;
    result.label = input_.label;

    auto& unit_volumes = csg_unit.tree.volumes();
    // Save unit's bounding box
    {
        NodeId node_id = unit_volumes[orange_exterior_volume.get()];
        auto region_iter = csg_unit.regions.find(node_id);
        CELER_ASSERT(region_iter != csg_unit.regions.end());
        auto const& bz = region_iter->second.bounds;
        if (bz.negated)
        {
            // [EXTERIOR] bbox is negated, so negating again gives the
            // "interior" bounding zone; we want its outer boundary.
            result.bbox = bz.exterior;
        }
        CELER_ENSURE(is_finite(result.bbox));
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
    detail::InternalSurfaceFlagger has_internal_surfaces{csg_unit.tree};
    result.volumes.reserve(unit_volumes.size()
                           + static_cast<bool>(csg_unit.background));

    for (auto vol_idx : range(unit_volumes.size()))
    {
        NodeId node_id = unit_volumes[vol_idx];
        VolumeInput vi;

        // Construct logic and faces with remapped surfaces
        auto&& [faces, logic] = detail::build_logic(
            detail::PostfixBuildLogicPolicy{csg_unit.tree,
                                            sorted_local_surfaces},
            node_id);
        vi.faces = std::move(faces);
        vi.logic = std::move(logic);
        // Set bounding box
        auto region_iter = csg_unit.regions.find(node_id);
        CELER_ASSERT(region_iter != csg_unit.regions.end());
        vi.bbox = get_exterior_bbox(region_iter->second.bounds);

        // Set bounding zone
        vi.obz.inner = region_iter->second.bounds.interior;
        vi.obz.outer = region_iter->second.bounds.exterior;
        vi.obz.trans_id = region_iter->second.trans_id;

        /*!
         * \todo "simple safety" flag is set inside "unit inserter": move here
         * once we stop importing from SCALE via OrangeInput.
         */
        if (has_internal_surfaces(node_id))
        {
            vi.flags |= VolumeRecord::internal_surfaces;
        }

        vi.zorder = ZOrder::media;
        result.volumes.emplace_back(std::move(vi));
    }

    if (csg_unit.background)
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
        /*! \todo The nearest internal surface is probably *not* the safety
         * distance, but it's better than nothing
         */
        vi.flags = VolumeRecord::implicit_vol | VolumeRecord::simple_safety;
        result.volumes.emplace_back(std::move(vi));
    }
    CELER_ASSERT(result.volumes.size()
                 == unit_volumes.size()
                        + static_cast<bool>(csg_unit.background));

    // Set labels and other attributes.
    // NOTE: this means we're entirely ignoring the "metadata" from the CSG
    // nodes for the region, because we can't know which ones have the
    // user-supplied volume names
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
    vol_iter->label = Label{"[EXTERIOR]", input_.label.name};
    ++vol_iter;

    BoundingBoxBumper<real_type> bump_bbox{input.tol()};
    for (auto const& d : input_.daughters)
    {
        LocalVolumeId const vol_id{
            static_cast<size_type>(vol_iter - result.volumes.begin())};

        // Save daughter attributes
        vol_iter->label = d.label;
        if (vol_iter->label == VariantLabel{})
        {
            // Choose default label for the volume
            vol_iter->label = Label{std::string{d.fill->label()},
                                    std::string{this->label()}};
        }
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
        auto trans_id = fill->trans_id;
        CELER_ASSERT(trans_id < csg_unit.transforms.size());
        iter->second.transform = csg_unit.transforms[trans_id.get()];

        // Update bounding box of the daughter universe by inverting the
        // daughter-to-parent reference transform and applying it to the
        // parent-reference-frame bbox
        auto local_bbox = apply_transform(calc_inverse(iter->second.transform),
                                          result.volumes[vol_id.get()].bbox);
        input.expand_bbox(iter->second.universe_id, bump_bbox(local_bbox));
    }

    // Save attributes from materials
    for (auto const& m : input_.materials)
    {
        CELER_ASSERT(vol_iter != result.volumes.end());
        vol_iter->label = m.label;
        if (vol_iter->label == decltype(m.label){})
        {
            // Default: empty label
            vol_iter->label = Label{std::string(m.interior->label())};
        }
        vol_iter->zorder = ZOrder::media;
        ++vol_iter;
    }

    if (auto const& b = input_.background)
    {
        CELER_ASSERT(vol_iter != result.volumes.end());
        vol_iter->label = b.label;
        ++vol_iter;
    }
    CELER_EXPECT(vol_iter == result.volumes.end());

    if (input.save_json())
    {
        // Write debug information
        JsonPimpl jp;
        jp.obj = csg_unit;
        jp.obj["remapped_surfaces"] = [&sorted_local_surfaces] {
            auto j = nlohmann::json::array();
            for (auto const& lsid : sorted_local_surfaces)
            {
                j.push_back(lsid.unchecked_get());
            }
            return j;
        }();

        // Save label volumes
        CELER_ASSERT(jp.obj.contains("volumes"));
        auto& jv = jp.obj["volumes"];
        CELER_VALIDATE(jv.size() == unit_volumes.size(),
                       << "jv = " << jv.size()
                       << " csg = " << unit_volumes.size());
        CELER_ASSERT(unit_volumes.size() <= result.volumes.size());
        for (auto vol_idx : range(unit_volumes.size()))
        {
            jv[vol_idx]["label"]
                = std::visit([](auto&& obj) -> nlohmann::json { return obj; },
                             result.volumes[vol_idx].label);
        }

        // Save our universe label
        jp.obj["label"] = this->label();

        // Update daughter universe IDs
        for (auto& v : jp.obj["volumes"])
        {
            if (auto iter = v.find("universe"); iter != v.end())
            {
                // The "universe" key is set with `fill_volume` in `build`
                // below as the daughter index
                std::size_t daughter_index = iter->get<int>();
                CELER_ASSERT(daughter_index < input_.daughters.size());
                auto const& daughter = input_.daughters[daughter_index];
                auto univ_id = input.find_universe_id(daughter.fill.get());
                *iter = univ_id.unchecked_get();
            }
        }

        input.save_json(std::move(jp));
    }

    //! \todo Save material IDs as well
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
auto UnitProto::build(Tol const& tol, BBox const& bbox) const -> Unit
{
    CELER_EXPECT(tol);
    CELER_EXPECT(!bbox || is_finite(bbox));

    bool const is_global_universe = !static_cast<bool>(bbox);
    CELER_LOG(debug) << "Building '" << this->label() << "' inside " << bbox
                     << ": " << input_.daughters.size() << " daughters and "
                     << input_.materials.size() << " materials...";

    detail::CsgUnit result;
    detail::CsgUnitBuilder unit_builder(
        &result, tol, is_global_universe ? BBox::from_infinite() : bbox);

    auto build_volume = [ub = &unit_builder](ObjectInterface const& obj) {
        detail::VolumeBuilder vb{ub};
        auto final_node = obj.build(vb);
        return ub->insert_volume(final_node);
    };

    // Build exterior volume and optional background fill
    if (input_.boundary.zorder != ZOrder::media && !input_.background)
    {
        CELER_NOT_IMPLEMENTED("implicit exterior without background fill");
    }
    auto ext_vol
        = build_volume(NegatedObject("[EXTERIOR]", input_.boundary.interior));
    CELER_ASSERT(ext_vol == orange_exterior_volume);
    if (is_global_universe)
    {
        detail::VolumeBuilder vb{&unit_builder};
        auto interior_node = input_.boundary.interior->build(vb);
        auto region_iter = result.regions.find(interior_node);
        CELER_ASSERT(region_iter != result.regions.end());
        auto const& bz = region_iter->second.bounds;
        CELER_VALIDATE(!bz.negated && is_finite(bz.exterior),
                       << "global boundary must be finite: cannot determine "
                          "extents of interior '"
                       << input_.boundary.interior->label() << "' in '"
                       << this->label() << '\'');
    }

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
        if (m.fill)
        {
            unit_builder.fill_volume(lv, m.fill);
        }
    }

    // Build background fill (optional)
    result.background = input_.background.fill;

    if (!is_global_universe)
    {
        // Replace "exterior" with "False" (i.e. interior with true)
        NodeId ext_node = result.tree.volumes()[ext_vol.unchecked_get()];
        auto unknowns = replace_and_simplify(&result.tree, ext_node, False{});
        if (!unknowns.empty())
        {
            auto write_node_labels = [&md = result.metadata](std::ostream& os,
                                                             NodeId nid) {
                CELER_ASSERT(nid < md.size());
                auto const& labels = md[nid.get()];
                os << '{' << join(labels.begin(), labels.end(), ", ") << '}';
            };
            CELER_LOG(warning)
                << "While building '" << this->label()
                << "', encountered surfaces that could not be logically "
                   "eliminated from the boundary: "
                << join_stream(unknowns.begin(),
                               unknowns.end(),
                               ", ",
                               write_node_labels);
        }

        if (input_.simplification == UnitSimplification::infix_logic)
        {
            unit_builder.simplifiy_joins();
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
void UnitProto::output(JsonPimpl* j) const
{
    using json = nlohmann::json;

    auto obj = json::object({{"label", input_.label}});
    VolumeToString to_string;

    if (auto& bg = input_.background)
    {
        obj["background"] = {
            {"fill", bg.fill.unchecked_get()},
            {"label", std::visit(to_string, bg.label)},
        };
    }

    obj["materials"] = [&ms = input_.materials, to_string] {
        auto result = json::array();
        for (auto const& m : ms)
        {
            result.push_back({
                {"interior", m.interior},
                {"fill", m.fill.get()},
                {"label", std::visit(to_string, m.label)},
            });
        }
        return result;
    }();

    obj["daughters"] = [&ds = input_.daughters, to_string] {
        auto result = json::array();
        for (auto const& d : ds)
        {
            result.push_back({
                {"fill", d.fill->label()},
                {"transform", d.transform},
                {"zorder", to_cstring(d.zorder)},
                {"label", std::visit(to_string, d.label)},
            });
        }
        return result;
    }();

    obj["boundary"] = {
        {"interior", input_.boundary.interior},
        {"zorder", to_cstring(input_.boundary.zorder)},
    };

    j->obj = std::move(obj);
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
