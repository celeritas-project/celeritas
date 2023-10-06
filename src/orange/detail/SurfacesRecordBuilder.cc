//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/SurfacesRecordBuilder.cc
//---------------------------------------------------------------------------//
#include "SurfacesRecordBuilder.hh"

#include <variant>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with pointers to the underlying storage.
 */
SurfacesRecordBuilder::SurfacesRecordBuilder(Items<SurfaceType>* types,
                                             Items<RealId>* real_ids,
                                             Items<real_type>* reals)
    : types_{types}, real_ids_{real_ids}, reals_{reals}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct a record of all the given surfaces.
 */
auto SurfacesRecordBuilder::operator()(VecSurface const& surfaces)
    -> result_type
{
    types_.reserve(types_.size() + surfaces.size());
    real_ids_.reserve(real_ids_.size() + surfaces.size());

    // Starting index for types and IDs
    auto begin_types = types_.size_id();
    auto begin_real_ids = real_ids_.size_id();

    // Functor to save the surface type and data, and the data offset
    auto emplace_surface = [this](auto&& s) {
        types_.push_back(s.surface_type());
        auto data = s.data();
        // TODO: if deduplicating, we can just give the range to an already
        // saved block of data
        auto real_range = reals_.insert_back(data.begin(), data.end());
        real_ids_.push_back(*real_range.begin());
    };

    // Save all surfaces
    for (auto const& s : surfaces)
    {
        CELER_ASSUME(!s.valueless_by_exception());
        std::visit(emplace_surface, s);
    }

    result_type result;
    result.surfaces.types = {begin_types, types_.size_id()};
    result.surfaces.data_offsets = {begin_real_ids, real_ids_.size_id()};
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
