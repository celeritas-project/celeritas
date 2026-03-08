//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AuxMockParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxParams.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"

#include "AuxMockData.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Mock class for shared host data that has associated thread-local data.
 */
class AuxMockParams : public AuxParams<AuxMockParamsData, AuxMockStateData>
{
  public:
    //!@{
    //! \name Type aliases
    using VecInt = std::vector<int>;
    //!@}

  public:
    // Construct with properties and IDs
    AuxMockParams(std::string&& label,
                  AuxId auxid,
                  int num_bins,
                  VecInt const& integers);

    //!@{
    //! \name Aux interface

    //! Short name for the data
    std::string_view label() const final { return label_; }
    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
    //!@}

    //!@{
    //! \name Data interface

    //! Access data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }
    //! Access data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }
    //!@}

    //! Access host/device params/state ref
    using AuxParams::ref;

  private:
    std::string label_;
    AuxId aux_id_;
    ParamsDataStore<AuxMockParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
