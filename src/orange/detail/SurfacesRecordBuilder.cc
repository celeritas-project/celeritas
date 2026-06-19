//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
        if constexpr (std::remove_reference_t<decltype(s)>::surface_type()
                      == SurfaceType::inv)
        {
            // See discussion on
            // https://github.com/celeritas-project/celeritas/pull/1342
            CELER_NOT_IMPLEMENTED("runtime involute support");
        }
        types_.push_back(s.surface_type());
        auto data = s.data();
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
    result.types = {begin_types, types_.size_id()};
    result.data_offsets = {begin_real_ids, real_ids_.size_id()};

    CELER_ENSURE(types_.size() == real_ids_.size());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
