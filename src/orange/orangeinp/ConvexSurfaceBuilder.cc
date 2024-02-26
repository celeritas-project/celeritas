//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ConvexSurfaceBuilder.cc
//---------------------------------------------------------------------------//
#include "ConvexSurfaceBuilder.hh"

#include "orange/BoundingBoxUtils.hh"
#include "orange/surf/RecursiveSimplifier.hh"
#include "orange/surf/SurfaceClipper.hh"

#include "detail/ConvexSurfaceState.hh"
#include "detail/CsgUnitBuilder.hh"
#include "detail/NegatedSurfaceClipper.hh"

namespace celeritas
{
namespace orangeinp
{
namespace
{
//---------------------------------------------------------------------------//
struct ClipImpl
{
    detail::BoundingZone* bzone{nullptr};

    template<class S>
    void operator()(Sense s, S const& surf)
    {
        if (s == Sense::inside)
        {
            SurfaceClipper clip{&bzone->interior, &bzone->exterior};
            clip(surf);
        }
        else
        {
            detail::NegatedSurfaceClipper clip{bzone};
            clip(surf);
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with persistent unit builder and less persistent state.
 *
 * Both arguments must have lifetimes that exceed the surface builder, but the
 * "unit builder" will have a duration of the whole unit construction, whereas
 * the state just has the duration of the convex surface set being built.
 */
ConvexSurfaceBuilder::ConvexSurfaceBuilder(UnitBuilder* ub, State* state)
    : ub_{ub}, state_{state}
{
    CELER_EXPECT(ub_ && state_);
    CELER_EXPECT(*state_);
}

//---------------------------------------------------------------------------//
/*!
 * Add a surface with a sense.
 *
 * The resulting surface *MUST* result in a convex region.
 */
template<class S>
void ConvexSurfaceBuilder::operator()(Sense sense, S const& surf)
{
    // First, clip the local bounding zone based on the given surface
    RecursiveSimplifier clip_simplified_local(ClipImpl{&state_->local_bzone},
                                              ub_->tol());
    clip_simplified_local(sense, surf);

    // Next, apply the transform and insert
    return this->insert_transformed(state_->make_face_name(sense, surf),
                                    sense,
                                    apply_transform(*state_->transform, surf));
}

//---------------------------------------------------------------------------//
// HELPER FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Add a surface after transforming it to an unknown type.
 *
 * \param extension Constructed metadata for the surface node
 * \param sense Whether the convex region is inside/outside this surface
 * \param surf Type-deleted surface
 */
void ConvexSurfaceBuilder::insert_transformed(std::string&& extension,
                                              Sense sense,
                                              VariantSurface const& surf)
{
    NodeId node_id;
    auto construct_impl = [&](Sense final_sense, auto&& final_surf) {
        using SurfaceT = std::decay_t<decltype(final_surf)>;

        // Insert transformed surface, deduplicating and creating CSG node
        node_id = ub_->insert_surface(final_surf);

        // Replace sense so we know to flip the node if needed
        sense = final_sense;

        // Update surface's global-reference bounding zone using *deduplicated*
        // surface
        ClipImpl{&state_->global_bzone}(final_sense,
                                        ub_->get_surface<SurfaceT>(node_id));
    };

    // Construct transformed surface, get back the node ID, update the sense
    RecursiveSimplifier construct_final(construct_impl, ub_->tol());
    construct_final(sense, surf);
    CELER_ASSERT(node_id);

    // Add metadata for the surface node
    ub_->insert_md(node_id, Label{state_->object_name, std::move(extension)});

    if (sense == Sense::inside)
    {
        // "Inside" the surface (negative quadric evaluation) means we have to
        // negate the CSG result
        static_assert(Sense::inside == to_sense(false));
        node_id = ub_->insert_csg(Negated{node_id});
    }

    // Add sense to "joined" region
    state_->nodes.push_back(node_id);
}

//---------------------------------------------------------------------------//
/*!
 * Shrink the exterior bounding boxes.
 */
void ConvexSurfaceBuilder::shrink_exterior(BBox const& bbox)
{
    CELER_EXPECT(bbox && !is_degenerate(bbox));

    {
        // Local
        BBox& exterior = state_->local_bzone.exterior;
        exterior = calc_intersection(exterior, bbox);
    }
    {
        // Global
        BBox& exterior = state_->global_bzone.exterior;
        exterior = calc_intersection(
            exterior, apply_transform(*state_->transform, bbox));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Grow the interior local bounding box.
 */
void ConvexSurfaceBuilder::grow_interior(BBox const& bbox)
{
    CELER_EXPECT(bbox);

    {
        // Local
        BBox& interior = state_->local_bzone.interior;
        interior = calc_union(interior, bbox);
    }
    {
        // Global
        BBox& interior = state_->global_bzone.interior;
        interior
            = calc_union(interior, apply_transform(*state_->transform, bbox));
    }
}

//---------------------------------------------------------------------------//
// FREE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply a convex surface builder to an unknown type.
 */
void visit(ConvexSurfaceBuilder& csb, Sense sense, VariantSurface const& surf)
{
    std::visit([&csb, sense](auto const& s) { csb(sense, s); }, surf);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATIONS
//---------------------------------------------------------------------------//
//! \cond
#define CSB_INSTANTIATE(SURF) \
    template void ConvexSurfaceBuilder::operator()(Sense, SURF const&);
CSB_INSTANTIATE(PlaneAligned<Axis::x>);
CSB_INSTANTIATE(PlaneAligned<Axis::y>);
CSB_INSTANTIATE(PlaneAligned<Axis::z>);
CSB_INSTANTIATE(CylCentered<Axis::x>);
CSB_INSTANTIATE(CylCentered<Axis::y>);
CSB_INSTANTIATE(CylCentered<Axis::z>);
CSB_INSTANTIATE(SphereCentered);
CSB_INSTANTIATE(CylAligned<Axis::x>);
CSB_INSTANTIATE(CylAligned<Axis::y>);
CSB_INSTANTIATE(CylAligned<Axis::z>);
CSB_INSTANTIATE(Plane);
CSB_INSTANTIATE(Sphere);
CSB_INSTANTIATE(ConeAligned<Axis::x>);
CSB_INSTANTIATE(ConeAligned<Axis::y>);
CSB_INSTANTIATE(ConeAligned<Axis::z>);
CSB_INSTANTIATE(SimpleQuadric);
CSB_INSTANTIATE(GeneralQuadric);
#undef CSB_INSTANTIATE
//! \endcond

}  // namespace orangeinp
}  // namespace celeritas
