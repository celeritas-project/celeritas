//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitParams.cc
//---------------------------------------------------------------------------//
#include "TrackInitParams.hh"

#include "base/CollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with primaries and storage factor.
 */
TrackInitParams::TrackInitParams(const Input& inp)
{
    CELER_EXPECT(!inp.primaries.empty());
    CELER_EXPECT(inp.storage_factor > 0);

    make_builder(&data_.primaries)
        .insert_back(inp.primaries.begin(), inp.primaries.end());
    data_.storage_factor = inp.storage_factor;
    host_ref_            = data_;

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
