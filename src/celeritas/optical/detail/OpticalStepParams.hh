//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalStepParams.hh
//---------------------------------------------------------------------------//
#pragma once
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxParams.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"

#include "OpticalStepData.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
// Reuse generic step state

/*!
 * Optical-specific step params.
 * Minimal implementation: post-step pos + volume only.
 */
class OpticalStepParams
    : public AuxParams<OpticalStepParamsData, OpticalStepStateData>
{
  public:
    OpticalStepParams(std::string&& label, AuxId id);

    std::string_view label() const final { return label_; }
    AuxId aux_id() const final { return aux_id_; }
    // Data interface
    HostRef const& host_ref() const final { return data_.host_ref(); }
    DeviceRef const& device_ref() const final { return data_.device_ref(); }
    std::vector<OpticalStepRecord>& records() { return records_; }

    // UPState create_state(MemSpace, StreamId, size_type) const final;

  private:
    std::string label_;
    AuxId aux_id_;

    ParamsDataStore<OpticalStepParamsData> data_;

    std::vector<OpticalStepRecord> records_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
