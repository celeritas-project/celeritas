//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/UserMockParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/UserInterface.hh"
#include "corecel/data/UserStateData.hh"

#include "UserMockData.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Mock class for shared host data that has associated thread-local data.
 */
class UserMockParams : public UserParamsInterface,
                       public ParamsDataInterface<UserMockParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using VecInt = std::vector<int>;
    template<MemSpace M>
    using StateT = UserStateData<UserMockStateData, M>;
    //!@}

  public:
    // Construct with properties and IDs
    UserMockParams(std::string&& label,
                   UserId uid,
                   int num_bins,
                   VecInt const& integers);

    //!@{
    //! \name User interface
    //! Short name for the data
    std::string_view label() const final { return label_; }
    //! Index of this class instance in its registry
    UserId user_id() const final { return user_id_; }
    // Build state data for a stream
    UPState create_state(MemSpace, StreamId, size_type) const final;
    //!@}

    //!@{
    //! \name Data interface
    //! Access data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }
    //! Access data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }
    //!@}

  private:
    std::string label_;
    UserId user_id_;
    CollectionMirror<UserMockParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
