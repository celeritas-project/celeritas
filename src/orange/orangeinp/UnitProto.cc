//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/UnitProto.cc
//---------------------------------------------------------------------------//
#include "UnitProto.hh"

#include <algorithm>

#include "CsgObject.hh"
#include "CsgTree.hh"
#include "CsgTreeUtils.hh"
#include "Transformed.hh"

#include "detail/CsgUnit.hh"
#include "detail/CsgUnitBuilder.hh"
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
void UnitProto::build(GlobalBuilder&) const
{
    // Transform CsgUnit to OrangeInput
    // - Map CSG nodes to volume IDs
    // - Map used CSG nodes to surface IDs
    // - Map universe IDs (index in daughter list to actual universe ID)
    // - Remap surface indices, removing unused surfaces
    // - Set up "interior" cell if needed (build volume and "all surfaces")
    // - Construct postfix logic definitions
    // - Copy bounding boxes
    CELER_NOT_IMPLEMENTED("global builder");
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
    CELER_ASSERT(ext_vol == local_orange_outside_volume);

    // Build daughters
    UniverseId daughter_id{0};
    for (auto const& d : input_.daughters)
    {
        if (d.zorder != ZOrder::media)
        {
            CELER_NOT_IMPLEMENTED("volume masking using different z orders");
        }
        auto lv = build_volume(*d.make_interior());
        unit_builder.fill_volume(lv, daughter_id++);
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
std::shared_ptr<Transformed> UnitProto::DaughterInput::make_interior() const
{
    CELER_EXPECT(*this);
    return std::make_shared<Transformed>(this->fill->interior(),
                                         VariantTransform{this->transform});
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
