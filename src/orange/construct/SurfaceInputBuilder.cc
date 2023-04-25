//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/SurfaceInputBuilder.cc
//---------------------------------------------------------------------------//
#include "SurfaceInputBuilder.hh"

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/io/Label.hh"

#include "OrangeInput.hh"

namespace celeritas
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
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to empty surfaces.
 */
SurfaceInputBuilder::SurfaceInputBuilder(SurfaceInput* input) : input_(input)
{
    CELER_EXPECT(input_ && input_->types.empty() && input_->sizes.empty()
                 && input_->data.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Insert a generic surface.
 */
LocalSurfaceId SurfaceInputBuilder::operator()(GenericSurfaceRef generic_surf,
                                               Label const& label)
{
    CELER_EXPECT(generic_surf);

    LocalSurfaceId::size_type new_id = input_->size();
    input_->types.push_back(generic_surf.type);
    input_->data.insert(
        input_->data.end(), generic_surf.data.begin(), generic_surf.data.end());
    input_->sizes.push_back(generic_surf.data.size());
    input_->labels.push_back(label);

    CELER_ENSURE(input_->types.size() == input_->sizes.size());
    CELER_ENSURE(input_->types.size() == input_->labels.size());
    return LocalSurfaceId{new_id};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
