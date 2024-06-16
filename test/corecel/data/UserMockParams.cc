//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/UserMockParams.cc
//---------------------------------------------------------------------------//
#include "UserMockParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct with properties.
 */
UserMockParams::UserMockParams(std::string&& label,
                               UserId uid,
                               int num_bins,
                               VecInt const& integers)
    : label_{std::move(label)}, user_id_{uid}
{
    CELER_EXPECT(user_id_);
    CELER_EXPECT(num_bins > 0);

    HostVal<UserMockParamsData> hp;
    hp.num_bins = num_bins;
    CollectionBuilder{&hp.integers}.insert_back(integers.begin(),
                                                integers.end());
    data_ = CollectionMirror{std::move(hp)};

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
auto UserMockParams::create_state(MemSpace memspace,
                                  StreamId stream,
                                  size_type size) const -> UPState
{
    CELER_EXPECT(stream);
    CELER_EXPECT(size > 0);

    return make_user_state<UserMockStateData>(*this, memspace, stream, size);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
