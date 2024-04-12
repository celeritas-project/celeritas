//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LocalSurfaceInserter.cc
//---------------------------------------------------------------------------//
#include "LocalSurfaceInserter.hh"

#include "SurfaceHashPoint.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace
{
real_type calc_length_scale(Tolerance<> const& tol)
{
    CELER_EXPECT(tol);
    return tol.abs / tol.rel;
}

}  // namespace
//---------------------------------------------------------------------------//
/*!
 * Construct with tolerance and a pointer to the surfaces vector.
 *
 * This builds an acceleration grid using the length scale of the problem
 * (deduced from the tolerance) and the relative tolerance.
 */
LocalSurfaceInserter::LocalSurfaceInserter(VecSurface* v,
                                           Tolerance<> const& tol)
    : surfaces_{v}
    , soft_surface_equal_{tol}
    , grid_{real_type{0.01} * calc_length_scale(tol), tol.rel}
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
    LocalSurfaceId const source_id(all_surf.size());

    // Get the surface corresponding to this ID
    auto get_surface = [&](LocalSurfaceId target_id) -> S const& {
        CELER_ASSERT(target_id < all_surf.size());
        VariantSurface const& target = all_surf[target_id.unchecked_get()];
        CELER_ASSUME(std::holds_alternative<S>(target));
        return std::get<S>(target);
    };

    // Hash the surface and get one or two possible iterators in the same bin
    // as the source surface
    auto inserted_iters = grid_.insert(
        S::surface_type(), SurfaceHashPoint{}(source), source_id);

    // Test for exact equality with all possible matches in this range
    LocalSurfaceId possible_match;
    LocalSurfaceId exact_match;
    auto test_equality = [&](LocalSurfaceId lsid) {
        // Get target surface
        S const& target = get_surface(lsid);

        // Test for soft equality
        if (soft_surface_equal_(source, target))
        {
            if (!possible_match)
            {
                // Save first matching surface
                possible_match = lsid;
            }
            // Test for exact equality
            if (exact_surface_equal_(source, target))
            {
                // Save match
                exact_match = lsid;
            }
        }
    };

    for (auto const& iter : {inserted_iters.first, inserted_iters.second})
    {
        // Test for soft equality against all possible surfaces with a nearby
        // "hash"
        for (auto&& [first, last] = grid_.equal_range(iter);
             first != last && !exact_match;
             ++first)
        {
            if (first != iter)
            {
                test_equality(first->second);
            }
        }
    }

    if (!possible_match)
    {
        // Surface is completely unique
        all_surf.emplace_back(std::in_place_type<S>, source);
        return source_id;
    }
    if (exact_match)
    {
        // Erase the possible surface from the grid hash
        grid_.erase(inserted_iters.first);
        if (inserted_iters.second != grid_.end())
        {
            grid_.erase(inserted_iters.second);
        }
        return exact_match;
    }

    // Surface is a little bit different, so we still need to insert it
    // to chain duplicates
    all_surf.emplace_back(std::in_place_type<S>, source);

    // Store the equivalency relationship and potentially chain equivalent
    // surfaces
    return this->merge_impl(source_id, possible_match);
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
