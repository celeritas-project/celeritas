//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LocalSurfaceInserter.cc
//---------------------------------------------------------------------------//
#include "LocalSurfaceInserter.hh"

#include <iostream>

#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayOperators.hh"
#include "orange/surf/SurfaceIO.hh"

#include "SurfaceHashPoint.hh"
using std::cout;
using std::endl;

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
real_type calc_length_scale(Tolerance<> const& tol)
{
    CELER_EXPECT(tol);
    return tol.abs / tol.rel;
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
    , calc_hashes_{real_type{0.01} * calc_length_scale(tol), 2 * tol.rel}
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

    cout << " * Adding surface " << source << endl;

    // Test for exact equality with all possible matches in this range
    LocalSurfaceId near_match;
    LocalSurfaceId exact_match;
    auto test_equality = [&](LocalSurfaceId lsid) {
        // Get target surface
        S const& target = get_surface(lsid);

        cout << "   + Testing against " << target << ": ";

        // Test for soft equality
        if (soft_surface_equal_(source, target))
        {
            if (!near_match)
            {
                // Save first matching surface
                near_match = lsid;
            }
            // Test for exact equality
            if (exact_surface_equal_(source, target))
            {
                // Save match
                exact_match = lsid;
                cout << "exact match";
            }
            else
            {
                auto delta = make_array(source.data());
                delta -= make_array(target.data());
                cout << "inexact match (delta=" << repr(delta) << ")";
            }
        }
        else
        {
            cout << "not soft equal";
        }
        cout << endl;
    };

    // Hash the surface and get keys for possible surfaces that might be
    // equivalent to the source
    auto possible_keys
        = calc_hashes_(S::surface_type(), SurfaceHashPoint{}(source));
    for (auto key : possible_keys)
    {
        if (key == SurfaceGridHash::redundant())
        {
            cout << "  - Skipping redundant key " << endl;
            continue;
        }
        cout << "  - Checking key " << key << endl;

        // Find possibly similar surfaces that match this key
        for (auto&& [iter, last] = hashed_surfaces_.equal_range(key);
             iter != last;
             ++iter)
        {
            test_equality(iter->second);

            if (exact_match)
            {
                // No need for further searching; we're identical
                cout << "  -> Returned exact match " << exact_match.get()
                     << endl;
                return exact_match;
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
        // Surface is eqivalent to an existing one but not identical: save and
        // return the deduplicated surface
        auto result = this->merge_impl(source_id, near_match);
        cout << "  -> Merged soft equal surface " << source_id.get() << " => "
             << near_match.get() << endl;
        return result;
    }

    cout << "  -> Inserted new surface " << source_id.get() << endl;
    return source_id;
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
