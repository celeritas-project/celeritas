//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LocalSurfaceInserter.cc
//---------------------------------------------------------------------------//
#include "LocalSurfaceInserter.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
LocalSurfaceInserter::LocalSurfaceInserter(VecSurface* v,
                                           Tolerance<> const& tol)
    : surfaces_{v}, soft_surface_equal_{tol}
{
    CELER_EXPECT(surfaces_);
    CELER_EXPECT(surfaces_->empty());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a surface with deduplication.
 */
template<class S>
LocalSurfaceId LocalSurfaceInserter::operator()(S const& source)
{
    VecSurface& all_surf = *surfaces_;
    VecSurfId& all_surf_ids = types_[S::surface_type()];

    auto get_surface = [&](LocalSurfaceId target_id) -> S const& {
        CELER_ASSERT(target_id < all_surf.size());
        VariantSurface const& target = all_surf[target_id.unchecked_get()];
        CELER_ASSUME(std::holds_alternative<S>(target));
        return std::get<S>(target);
    };

    // Test for soft equality against all existing surfaces
    // TODO: instead of linear search (making overall unit insertion
    // quadratic!) accelerate by mapping surfaces into a hash and comparing all
    // neighboring hash cells
    auto iter = std::find_if(
        all_surf_ids.begin(), all_surf_ids.end(), [&](LocalSurfaceId target_id) {
            return soft_surface_equal_(source, get_surface(target_id));
        });
    if (iter == all_surf_ids.end())
    {
        // Surface is completely unique
        LocalSurfaceId result(surfaces_->size());
        all_surf.emplace_back(std::in_place_type<S>, source);
        all_surf_ids.push_back(result);
        return result;
    }

    // Surface is soft equivalent to an existing surface at this index
    LocalSurfaceId target_id{*iter};
    if (exact_surface_equal_(source, get_surface(target_id)))
    {
        // Surfaces are *exactly* equal: don't insert
        return target_id;
    }

    // Surface is a little bit different, so we still need to insert it
    // to chain duplicates
    LocalSurfaceId source_id(all_surf.size());
    all_surf.emplace_back(std::in_place_type<S>, source);
    all_surf_ids.push_back(source_id);

    // Store the equivalency relationship and potentially chain equivalent
    // surfaces
    return this->merge_impl(source_id, target_id);
}

//---------------------------------------------------------------------------//
/*!
 * Look for duplicate surfaces and store equivalency relationship.
 */
LocalSurfaceId
LocalSurfaceInserter::merge_impl(LocalSurfaceId source, LocalSurfaceId target)
{
    CELER_EXPECT(source < surfaces_->size());
    CELER_EXPECT(target < source);

    if (auto iter = merged_.find(target); iter != merged_.end())
    {
        // Surface was already merged with another: chain them
        target = iter->second;
        CELER_ENSURE(target < source);
    }

    merged_.emplace(source, target);
    return target;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATIONS
//---------------------------------------------------------------------------//
//! \cond
#define LSI_INSTANTIATE(SURF) \
    template LocalSurfaceId LocalSurfaceInserter::operator()(SURF const&)

LSI_INSTANTIATE(PlaneAligned<Axis::x>);
LSI_INSTANTIATE(PlaneAligned<Axis::y>);
LSI_INSTANTIATE(PlaneAligned<Axis::z>);
LSI_INSTANTIATE(CylCentered<Axis::x>);
LSI_INSTANTIATE(CylCentered<Axis::y>);
LSI_INSTANTIATE(CylCentered<Axis::z>);
LSI_INSTANTIATE(SphereCentered);
LSI_INSTANTIATE(CylAligned<Axis::x>);
LSI_INSTANTIATE(CylAligned<Axis::y>);
LSI_INSTANTIATE(CylAligned<Axis::z>);
LSI_INSTANTIATE(Plane);
LSI_INSTANTIATE(Sphere);
LSI_INSTANTIATE(ConeAligned<Axis::x>);
LSI_INSTANTIATE(ConeAligned<Axis::y>);
LSI_INSTANTIATE(ConeAligned<Axis::z>);
LSI_INSTANTIATE(SimpleQuadric);
LSI_INSTANTIATE(GeneralQuadric);

#undef LSI_INSTANTIATE
//! \endcond

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
