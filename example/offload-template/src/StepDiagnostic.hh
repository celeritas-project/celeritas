//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <celeritas/global/ActionInterface.hh>
#include <corecel/data/AuxInterface.hh>
#include <corecel/data/CollectionMirror.hh>
#include <corecel/data/ParamsDataInterface.hh>

#include "StepDiagnosticData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreStateInterface;

namespace example
{
//---------------------------------------------------------------------------//
//! Statistics integrated over an event
struct StepStatistics
{
    using real_type = double;
    using size_type = unsigned long long;

    double step_length{};  // [mm]
    double energy_deposition{};  // [MeV]
    size_type num_steps{};
    size_type num_primaries{};
    size_type num_secondaries{};
};

//---------------------------------------------------------------------------//
/*!
 * Accumulate step diagnostics.
 *
 * This class is mostly boilerplate that in the future will be abstracted. It
 * manages "thread-local" (i.e., per-stream state auxiliary) data, launches
 * kernels to gather statistics, and provides an accessor for copying back to
 * the user regardless of where the accumulated data lives.
 */
class StepDiagnostic final : public CoreStepActionInterface,
                             public AuxParamsInterface,
                             public ParamsDataInterface<StepParamsData>
{
  public:
    // Construct and add to core params
    static std::shared_ptr<StepDiagnostic>
    make_and_insert(CoreParams const& core);

    // Construct with IDs and filename base
    StepDiagnostic(ActionId action_id, AuxId aux_id);

    //// USER INTEGRATION ////

    // Get the statistics and reset them
    StepStatistics GetAndReset(CoreStateInterface& state) const;

    //// CELERITAS INTEGRATION ////

    //!@{
    //! \name Metadata interface

    //! Label for the auxiliary data and action
    std::string_view label() const final { return sad_.label(); }
    // Description of the action
    std::string_view description() const final { return sad_.description(); }
    //!@}

    //!@{
    //! \name Step action interface

    //! Index of this class instance in its registry
    ActionId action_id() const final { return sad_.action_id(); }
    //! Index of this class instance in its registry
    StepActionOrder order() const final { return StepActionOrder::user_post; }
    // Execute the action with host data
    void step(CoreParams const& params, CoreStateHost& state) const final;
    // Execute the action with device data
    void step(CoreParams const& params, CoreStateDevice& state) const final;
    //!@}

    //!@{
    //! \name Aux params interface

    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
    // Build state data for a stream
    UPState create_state(MemSpace m, StreamId id, size_type size) const final;
    //!@}

    //!@{
    //! \name Data interface

    //! Access physics properties on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }
    //! Access physics properties on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }
    //!@}

  private:
    //// DATA ////

    StaticActionData sad_;
    AuxId aux_id_;
    CollectionMirror<StepParamsData> mirror_;
};

//---------------------------------------------------------------------------//
}  // namespace example
}  // namespace celeritas
