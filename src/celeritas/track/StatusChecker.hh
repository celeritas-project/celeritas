//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/StatusChecker.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

#include "corecel/Types.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/Types.hh"

#include "StatusCheckData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;
class CoreParams;
template<MemSpace>
class CoreState;

//---------------------------------------------------------------------------//
/*!
 * Verify a consistent simulation state after performing an action.
 *
 * This is used as a debug option in the step executor to check that actions
 * and simulation state are consistent.
 *
 * Since this is called manually by the stepper, multiple times per step, it is
 * \em not an "explicit" action.
 */
class StatusChecker final : public AuxParamsInterface,
                            public ParamsDataInterface<StatusCheckParamsData>
{
  public:
    // Construct with aux ID and action registry
    StatusChecker(AuxId aux_id, ActionRegistry const& registry);

    //!@{
    //! \name Aux interface
    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }

    //! Label for the auxiliary data
    std::string_view label() const final { return "status-check"; }

    // Build state data for a stream
    UPState create_state(MemSpace m, StreamId id, size_type size) const final;
    //!@}

    //!@{
    //! \name Data interface
    //! Access data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }
    //! Access data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }
    //!@}

    // Execute *manually* with the last action's ID and the state
    template<MemSpace M>
    void
    execute(ActionId prev_action, CoreParams const&, CoreState<M>& state) const;

  private:
    template<MemSpace M>
    using StatusStateRef = StatusCheckStateData<Ownership::reference, M>;

    //// DATA ////

    AuxId aux_id_;
    CollectionMirror<StatusCheckParamsData> data_;

    //// HELPER FUNCTIONS ////

    void launch_impl(CoreParams const&,
                     CoreState<MemSpace::host>&,
                     StatusStateRef<MemSpace::host> const&) const;
    void launch_impl(CoreParams const&,
                     CoreState<MemSpace::device>&,
                     StatusStateRef<MemSpace::device> const&) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
