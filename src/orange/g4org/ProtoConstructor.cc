//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/ProtoConstructor.cc
//---------------------------------------------------------------------------//
#include "ProtoConstructor.hh"

#include <iostream>

#include "corecel/OpaqueIdIO.hh"
#include "corecel/io/StreamableVariant.hh"
#include "geocel/VolumeParams.hh"
#include "orange/orangeinp/CsgObject.hh"
#include "orange/orangeinp/PolySolid.hh"
#include "orange/orangeinp/Transformed.hh"
#include "orange/transform/TransformIO.hh"

#include "Volume.hh"

namespace celeritas
{
namespace g4org
{
namespace
{
//---------------------------------------------------------------------------//
bool is_union(ProtoConstructor::SPConstObject const& obj)
{
    return dynamic_cast<orangeinp::AnyObjects const*>(obj.get())
           || dynamic_cast<orangeinp::PolySolidBase const*>(obj.get());
}

//---------------------------------------------------------------------------//
// It's possible to have Geant4 geometry with no materials defined:
// in this case, the background/material will be 'false' unless we use a
// bogus material ID
GeoMatId background_fill(GeoMatId mat)
{
    if (mat)
        return mat;
    return GeoMatId{0};
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct a proto-universe from a physical volume.
 *
 * We can use logical volume for the structure, but we need to associate the
 * world physical volume ID.
 */
auto ProtoConstructor::operator()(LogicalVolume const& lv) -> SPUnitProto
{
    auto const& label = volumes_.volume_labels().at(lv.id);

    UnitProto::Input input;
    input.boundary.interior = lv.solid;
    input.label = label;

    if (CELER_UNLIKELY(verbose_))
    {
        std::clog << std::string(depth_, ' ') << "* New proto: '" << label
                  << "' with shape " << to_string(*lv.solid) << std::endl;
    }

    // Add children
    for (PhysicalVolume const& child_pv : lv.children)
    {
        ++depth_;
        this->place_pv(NoTransformation{}, child_pv, &input);
        --depth_;
    }

    // Heuristic: if LV has fewer than N daughters in the input, use an
    // explicit background cell
    if (lv.children.size() <= fill_daughter_threshold())
    {
        // Create explicit "fill" impl volume for this logical volume. The
        // special "empty volume instance" label indicates to the
        // find_bg_volume function in Converter.cc that this needs to be set
        // as an LV rather than a PV.
        orangeinp::UnitProto::MaterialInput background;
        background.interior
            = this->make_explicit_background(lv, NoTransformation{});
        background.label = VolumeInstanceId{};
        background.fill = background_fill(lv.material_id);
        input.boundary.zorder = ZOrder::media;
        input.materials.push_back(std::move(background));

        if (CELER_UNLIKELY(verbose_))
        {
            std::clog << std::string(depth_, ' ') << " - explicit background "
                      << " for proto '" << label << "'" << std::endl;
        }
    }
    else
    {
        // Children are *implicitly* subtracted from the parent. Since this
        // universal fill corresponds to a *volume* rather than an *instance*,
        // we indicate this fill with a null volume instance ID.
        input.background.fill = background_fill(lv.material_id);
        input.background.label = VolumeInstanceId{};
        CELER_ASSERT(input.background);

        if (CELER_UNLIKELY(verbose_))
        {
            std::clog << std::string(depth_, ' ') << " - implicit background"
                      << " for proto '" << label << "'" << std::endl;
        }
    }

    CELER_ENSURE(input);
    return std::make_shared<orangeinp::UnitProto>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Place this physical volume into a proto being constructed.
 *
 * It will return a "local child ID" if it generates a material input, but not
 * if it spawns a daughter proto.
 */
void ProtoConstructor::place_pv(VariantTransform const& parent_transform,
                                PhysicalVolume const& pv,
                                UnitProto::Input* proto)
{
    CELER_EXPECT(proto);

    using namespace orangeinp;

    // Transform for this PV, whether as a "top level" volume or as a volume
    // that's subtracted from an inlined LV
    auto transform = apply_transform(parent_transform, pv.transform);

    if (CELER_UNLIKELY(verbose_))
    {
        std::clog << std::string(depth_, ' ') << "- Add pv '"
                  << volumes_.volume_instance_labels().at(pv.id)
                  << "' use_count=" << pv.lv.use_count()
                  << ", num_children=" << pv.lv->children.size() << ", at "
                  << StreamableVariant{transform} << " to '" << proto->label
                  << "'" << std::endl;
    }

    // Track relationship between this volume instance and embedded children
    auto add_material = [&](SPConstObject&& obj) {
        CELER_EXPECT(obj);
        UnitProto::MaterialInput mat;
        mat.interior = std::move(obj);
        mat.fill = pv.lv->material_id;
        mat.label = pv.id;
        proto->materials.push_back(std::move(mat));
    };

    if (pv.lv->children.empty())
    {
        // No children! This LV is just a material.
        add_material(
            Transformed::or_object(pv.lv->solid, std::move(transform)));

        if (CELER_UNLIKELY(verbose_))
        {
            std::clog << std::string(depth_, ' ') << " -> "
                      << "material at " << StreamableVariant{pv.transform}
                      << std::endl;
        }
    }
    else if ((pv.lv.use_count() == 1
              && std::holds_alternative<NoTransformation>(pv.transform))
             || is_union(pv.lv->solid))
    {
        // Child can be inlined into the parent because it's used only once
        // *and* it doesn't have a rotation relative to the parent
        // OR: it must be inlined if it's a union (see #1260)
        add_material(this->make_explicit_background(*pv.lv, transform));

        if (CELER_UNLIKELY(verbose_))
        {
            std::clog << std::string(depth_, ' ') << " -> "
                      << "inlined child to material at "
                      << StreamableVariant{pv.transform} << "; subtracting "
                      << pv.lv->children.size() << " children" << std::endl;
        }

        // Now build its children
        ++depth_;
        for (auto const& child_pv : pv.lv->children)
        {
            // Note: place_pv incorporates child's transform
            this->place_pv(transform, child_pv, proto);
        }
        --depth_;
    }
    else
    {
        // LV is referenced more than once *AND* has children *AND* has a
        // transform *BUT* is not a union
        auto [iter, inserted] = protos_.emplace(pv.lv.get(), nullptr);

        if (CELER_UNLIKELY(verbose_))
        {
            std::clog << std::string(depth_, ' ') << " -> "
                      << "placing " << (inserted ? "new" : "existing")
                      << " universe '"
                      << volumes_.volume_labels().at(pv.lv->id) << "' at "
                      << StreamableVariant{pv.transform} << std::endl;
        }

        if (inserted)
        {
            ++depth_;
            // Construct volume as a proto
            iter->second = (*this)(*pv.lv);
            --depth_;
        }
        CELER_ASSERT(iter->second);

        UnitProto::DaughterInput daughter;
        daughter.fill = iter->second;
        daughter.transform = transform;
        daughter.zorder = ZOrder::media;
        daughter.label = pv.id;
        proto->daughters.push_back(std::move(daughter));

        if (CELER_UNLIKELY(verbose_))
        {
            std::clog << std::string(depth_, ' ') << " :  "
                      << "daughter shape is "
                      << to_string(*proto->daughters.back().make_interior())
                      << std::endl;
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct an explicit "background" cell.
 *
 * This is the LV's volume with all of the direct daughter LVs subtracted.
 */
auto ProtoConstructor::make_explicit_background(
    LogicalVolume const& lv, VariantTransform const& transform) -> SPConstObject
{
    using namespace orangeinp;

    std::vector<SPConstObject> children;
    children.reserve(lv.children.size());
    for (auto const& child_pv : lv.children)
    {
        children.push_back(
            Transformed::or_object(child_pv.lv->solid, child_pv.transform));
    }

    SPConstObject interior;
    if (CELER_UNLIKELY(children.empty()))
    {
        // Rare case (world is the only volume!)
        return lv.solid;
    }
    else if (children.size() == 1)
    {
        // One child: interior becomes that object
        interior = std::move(children.front());
    }
    else
    {
        auto const& name = volumes_.volume_labels().at(lv.id).name;
        interior = std::make_shared<AnyObjects>(name + ".children",
                                                std::move(children));
    }

    auto const& name = volumes_.volume_labels().at(lv.id).name;
    return Transformed::or_object(
        orangeinp::make_subtraction(std::string{name}, lv.solid, interior),
        transform);
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
