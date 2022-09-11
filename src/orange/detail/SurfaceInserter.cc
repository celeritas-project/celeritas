//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/SurfaceInserter.cc
//---------------------------------------------------------------------------//
#include "SurfaceInserter.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "orange/construct/OrangeInput.hh"
#include "orange/surf/SurfaceAction.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
template<class T>
struct SurfaceDataSize
{
    constexpr size_type operator()() const noexcept
    {
        return T::Storage::extent;
    }
};

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to empty surfaces.
 */
SurfaceInserter::SurfaceInserter(Data* params) : orange_data_(params)
{
    CELER_EXPECT(orange_data_ && orange_data_->surfaces.types.empty()
                 && orange_data_->surfaces.offsets.empty()
                 && orange_data_->surfaces.reals.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Insert all surfaces at once.
 */
auto SurfaceInserter::operator()(const SurfaceInput& s) -> SurfaceRange
{
    //// Check input consistency ////

    CELER_VALIDATE(s.types.size() == s.sizes.size(),
                   << "inconsistent surfaces input: number of types ("
                   << s.types.size() << ") must match number of sizes ("
                   << s.sizes.size() << ")");

    auto get_data_size = make_static_surface_action<SurfaceDataSize>();

    size_type accum_size = 0;
    for (auto i : range(s.types.size()))
    {
        size_type expected_size = get_data_size(s.types[i]);
        CELER_VALIDATE(expected_size == s.sizes[i],
                       << "inconsistent surface data size (" << s.sizes[i]
                       << ") for entry " << i << ": "
                       << "surface type " << to_cstring(s.types[i])
                       << " should have " << expected_size);
        accum_size += expected_size;
    }

    CELER_VALIDATE(accum_size == s.data.size(),
                   << "incorrect surface data size (" << s.data.size()
                   << "): should match accumulated sizes (" << accum_size
                   << ")");

    //// Insert data ////

    SurfaceId start_id{orange_data_->surfaces.types.size()};
    size_type start_offset = orange_data_->surfaces.reals.size();

    auto types   = make_builder(&orange_data_->surfaces.types);
    auto offsets = make_builder(&orange_data_->surfaces.offsets);
    auto reals   = make_builder(&orange_data_->surfaces.reals);

    types.insert_back(s.types.begin(), s.types.end());
    reals.insert_back(s.data.begin(), s.data.end());

    offsets.reserve(offsets.size() + s.sizes.size());
    for (auto single_size : s.sizes)
    {
        offsets.push_back(OpaqueId<real_type>{start_offset});
        start_offset += single_size;
    }

    return {start_id, SurfaceId{types.size()}};
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
