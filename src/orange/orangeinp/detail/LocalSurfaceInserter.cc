//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
//---------------------------------------------------------------------------//
constexpr real_type calc_length_scale(Tolerance<> const& tol)
{
    CELER_EXPECT(tol);
    return tol.abs / tol.rel;
}

//---------------------------------------------------------------------------//
Tolerance<> scaled_tolerance(Tolerance<> const& tol, real_type scale)
{
    CELER_EXPECT(tol);
    CELER_EXPECT(scale > 0);
    Tolerance<> result;
    result.rel = tol.rel * scale;
    result.abs = tol.abs * scale;
    // Quietly correct too-small tolerances: only user construction should warn
    return result.clamped();
}

//---------------------------------------------------------------------------//
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
    , surface_equal_{scaled_tolerance(tol, exact_rel_tolerance_)}
    , calc_hashes_{bin_width_frac_ * calc_length_scale(tol), 2 * tol.abs}
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

    // Test for exact equality with all possible matches in this range
    LocalSurfaceId near_match;
    LocalSurfaceId exact_match;
    auto test_equality = [&](LocalSurfaceId lsid) {
        // Get target surface
        S const& target = get_surface(lsid);

        // Test for soft equality
        if (soft_surface_equal_(source, target))
        {
            if (!near_match)
            {
                // Save first matching surface
                near_match = lsid;
            }
            // Test for "exact" equality
            if (surface_equal_(source, target))
            {
                // Save match
                exact_match = lsid;
            }
        }
    };

    // Hash the surface and get keys for possible surfaces that might be
    // equivalent to the source
    auto possible_keys
        = calc_hashes_(S::surface_type(), SurfaceHashPoint{}(source));
    for (auto key : possible_keys)
    {
        if (key == SurfaceGridHash::redundant())
            continue;

        // Find possibly similar surfaces that match this key. The key
        // is *specific* to a surface type and *likely* the same spatial bin.
        for (auto&& [iter, last] = hashed_surfaces_.equal_range(key);
             iter != last;
             ++iter)
        {
            test_equality(iter->second);

            if (exact_match)
            {
                // No need for further searching; we're identical to an
                // existing surface (which may possibly be a "deduplicated"
                // copy of another surface)
                return this->find_merged(exact_match);
            }
        }
    }

    // Add the non-duplicate surface to the hashed surface list
    for (auto key : possible_keys)
    {
        if (key != SurfaceGridHash::redundant())
        {
            hashed_surfaces_.insert({key, source_id});
        }
    }

    // Add the non-exact surface even if it's a near match (since we need to
    // chain near-duplicates)
    all_surf.emplace_back(std::in_place_type<S>, source);

    if (near_match)
    {
        // Surface is equivalent to an existing one but not identical: save and
        // return the deduplicated surface
        return this->merge_impl(source_id, near_match);
    }

    return source_id;
}

//---------------------------------------------------------------------------//
/*!
 * Look for duplicate surfaces and store equivalency relationship.
 *
 * This marks the "target" (existing near match) as equivalent to the "source"
 * (new surface). It will chain merged surfaces so that multiple equivalent
 * ones will point to a single original.
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
/*!
 * Return an original chained target if one exists.
 *
 * This is necessary so that exact matches to a "soft" (deduplicated) surface
 * return the ID of an original surface rather than the exact match.
 */
LocalSurfaceId LocalSurfaceInserter::find_merged(LocalSurfaceId target) const
{
    CELER_EXPECT(target < surfaces_->size());
    if (auto iter = merged_.find(target); iter != merged_.end())
    {
        // Surface was already merged with another: chain them
        target = iter->second;
    }
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
LSI_INSTANTIATE(Involute);

#undef LSI_INSTANTIATE
//! \endcond

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
