//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OffloadParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/AuxInterface.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "../OffloadData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Manage metadata for optical offload generation.
 */
class OffloadParams final : public AuxParamsInterface,
                            public ParamsDataInterface<OffloadParamsData>
{
  public:
    // Construct with aux ID and optical data
    OffloadParams(AuxId aux_id, OffloadOptions const& setup);

    //!@{
    //! \name Aux interface
    //! Short name for the action
    std::string_view label() const final { return "optical-offload"; }
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
    CollectionMirror<OffloadParamsData> data_;
};

//---------------------------------------------------------------------------//
/*!
 * Manage optical generation states.
 */
template<MemSpace M>
struct OpticalOffloadState : public AuxStateInterface
{
    CollectionStateStore<OffloadStateData, M> store;
    OffloadBufferSize buffer_size;

    //! True if states have been allocated
    explicit operator bool() const { return static_cast<bool>(store); }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
