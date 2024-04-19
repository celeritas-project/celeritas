//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/ProtoConstructor.cc
//---------------------------------------------------------------------------//
#include "ProtoConstructor.hh"

#include <iostream>

#include "corecel/io/StreamableVariant.hh"
#include "orange/orangeinp/CsgObject.hh"
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
using SPConstObject = ProtoConstructor::SPConstObject;

//---------------------------------------------------------------------------//
/*!
 * Construct an explicit "background" cell.
 */
SPConstObject make_explicit_background(LogicalVolume const& lv,
                                       VariantTransform const& transform)
{
    using namespace orangeinp;

    std::vector<SPConstObject> children;
    for (auto const& child_pv : lv.children)
    {
        auto child_transform = apply_transform(transform, child_pv.transform);
        children.push_back(
            Transformed::or_object(child_pv.lv->solid, child_transform));
    }

    return Transformed::or_object(
        orangeinp::make_subtraction(
            std::string{lv.name},
            lv.solid,
            orangeinp::AnyObjects::or_object(lv.name + ".children",
                                             std::move(children))),
        transform);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct a proto-universe from a logical volume.
 */
auto ProtoConstructor::operator()(LogicalVolume const& lv) -> SPUnitProto
{
    ProtoInput input;
    input.boundary.interior = lv.solid;
    input.label = lv.name;

    if (CELER_UNLIKELY(verbose_))
    {
        std::clog << std::string(depth_, ' ') << "* New proto: " << lv.name
                  << " with shape " << to_string(*lv.solid) << std::endl;
    }

    // Add children first
    for (PhysicalVolume const& pv : lv.children)
    {
        ++depth_;
        this->place_pv(NoTransformation{}, pv, &input);
        --depth_;
    }

    // Heuristic: if LV has fewer than N daughters in the input, use an
    // explicit background cell
    if (lv.children.size() <= fill_daughter_threshold())
    {
        if (CELER_UNLIKELY(verbose_))
        {
            std::clog << std::string(depth_, ' ') << " - explicit background"
                      << std::endl;
        }
        // Create explicit "fill" for this logical volume
        orangeinp::UnitProto::MaterialInput background;
        background.interior = make_explicit_background(lv, NoTransformation{});
        background.label = Label::from_geant(lv.name);
        background.fill = lv.material_id;
        input.boundary.zorder = ZOrder::media;
        input.materials.push_back(std::move(background));
    }
    else
    {
        if (CELER_UNLIKELY(verbose_))
        {
            std::clog << std::string(depth_, ' ') << " - implicit background"
                      << std::endl;
        }
        input.background.fill = lv.material_id;
        input.background.label = Label::from_geant(lv.name);
    }

    CELER_ENSURE(input);
    return std::make_shared<orangeinp::UnitProto>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Place this physical volume into the proto.
 */
void ProtoConstructor::place_pv(VariantTransform const& parent_transform,
                                PhysicalVolume const& pv,
                                ProtoInput* proto)
{
    CELER_EXPECT(proto);

    using namespace orangeinp;

    // Transform for this PV, whether as a "top level" volume or as a volume
    // that's subtracted from an inlined LV
    auto transform = apply_transform(parent_transform, pv.transform);

    if (CELER_UNLIKELY(verbose_))
    {
        std::clog << std::string(depth_, ' ') << "- Add pv " << pv.name
                  << " use_count=" << pv.lv.use_count()
                  << ", num_children=" << pv.lv->children.size() << ", at "
                  << StreamableVariant{transform} << " to " << proto->label
                  << std::endl;
    }

    auto add_material = [proto, &lv = *pv.lv](SPConstObject&& obj) {
        CELER_EXPECT(obj);
        UnitProto::MaterialInput mat;
        mat.interior = std::move(obj);
        mat.fill = lv.material_id;
        mat.label = Label::from_geant(lv.name);
        proto->materials.push_back(std::move(mat));
    };

    if (pv.lv->children.empty())
    {
        // No children! This LV is just a material.
        if (CELER_UNLIKELY(verbose_))
        {
            std::clog << std::string(depth_, ' ') << " -> "
                      << "material at " << StreamableVariant{pv.transform}
                      << std::endl;
            std::clog << std::string(depth_, ' ') << "    "
                      << to_string(*pv.lv->solid) << std::endl;
        }

        add_material(
            Transformed::or_object(pv.lv->solid, std::move(transform)));
    }
    else if (pv.lv.use_count() == 1
             && std::holds_alternative<NoTransformation>(pv.transform))
    {
        // Child can be inlined into the parent because it's used only once
        // *and* it doesn't have a transform relative to the parent
        if (CELER_UNLIKELY(verbose_))
        {
            std::clog << std::string(depth_, ' ') << " -> "
                      << "inline the child at "
                      << StreamableVariant{pv.transform} << std::endl;
        }

        // Subtract *its* children from this shape
        if (CELER_UNLIKELY(verbose_))
        {
            std::clog << std::string(depth_, ' ') << " :  subtracted "
                      << pv.lv->children.size() << " children from "
                      << to_string(*pv.lv->solid) << std::endl;
        }

        add_material(make_explicit_background(*pv.lv, transform));

        // Now build its daghters
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
        // LV is referenced more than once *AND* has children
        if (CELER_UNLIKELY(verbose_))
        {
            std::clog << std::string(depth_, ' ') << " -> "
                      << "new universe at " << StreamableVariant{pv.transform}
                      << std::endl;
        }

        auto [iter, inserted] = protos_.insert({pv.lv.get(), nullptr});
        if (inserted)
        {
            ++depth_;
            // Construct pv proto
            iter->second = (*this)(*pv.lv);
            --depth_;
        }
        CELER_ASSERT(iter->second);
        proto->daughters.push_back({iter->second, transform, ZOrder::media});

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
}  // namespace g4org
}  // namespace celeritas
