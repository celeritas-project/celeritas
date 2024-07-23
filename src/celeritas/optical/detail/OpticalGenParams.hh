//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalGenParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/AuxInterface.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "../OpticalGenData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Manage metadata about optical generation.
 */
class OpticalGenParams final : public AuxParamsInterface,
                               public ParamsDataInterface<OpticalGenParamsData>
{
  public:
    // Construct with aux ID and optical data
    OpticalGenParams(AuxId aux_id, OpticalGenSetup const& setup);

    //!@{
    //! \name Aux interface
    //! Short name for the action
    std::string_view label() const final { return "optical-gen"; }
    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
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
    AuxId aux_id_;
    CollectionMirror<OpticalGenParamsData> data_;
};

//---------------------------------------------------------------------------//
/*!
 * Manage optical generation states.
 */
template<MemSpace M>
struct OpticalGenState : public AuxStateInterface
{
    CollectionStateStore<OpticalGenStateData, M> store;
    OpticalBufferSize buffer_size;

    //! True if states have been allocated
    explicit operator bool() const { return static_cast<bool>(store); }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
