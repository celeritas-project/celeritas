//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
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
    CELER_EXPECT(inp.capacity > 0);

    make_builder(&host_value_.primaries)
        .insert_back(inp.primaries.begin(), inp.primaries.end());
    host_value_.capacity       = inp.capacity;
    host_ref_                  = host_value_;

    CELER_ENSURE(host_value_);
    CELER_ENSURE(host_ref_);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
