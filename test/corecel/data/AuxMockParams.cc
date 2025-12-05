//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AuxMockParams.cc
//---------------------------------------------------------------------------//
#include "AuxMockParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct with properties.
 */
AuxMockParams::AuxMockParams(std::string&& label,
                             AuxId auxid,
                             int num_bins,
                             VecInt const& integers)
    : label_{std::move(label)}, aux_id_{auxid}
{
    CELER_EXPECT(aux_id_);
    CELER_EXPECT(num_bins > 0);

    HostVal<AuxMockParamsData> hp;
    hp.num_bins = num_bins;
    CollectionBuilder{&hp.integers}.insert_back(integers.begin(),
                                                integers.end());
    data_ = CollectionMirror{std::move(hp)};

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
