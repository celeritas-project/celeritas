//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/StepDiagnostic.hh
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
class CoreStateCounters;

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
 *
 * It is constructed by the user options (see \c MakeCelerOptions) and
 * integrates into the Geant4 transport loop in the user \c EventAction by
 * calling \c GetAndReset.
 *
 * To be consistent in Geant4, additional \c SteppingAction and \c
 * TrackingAction classes should be created to gather equivalent data from
 * Geant4.
 *
 * The StepDiagnostic inherits from three Celeritas classes that provide
 * interfaces:
 * - \c ParamsData provides a unified template interface for shared problem
 *   setup data.
 * - \c CoreStepActionInterface allows the class to be called at every step
 *   iteration with thread-local particle state data.
 * - \c AuxParamsInterface is needed to store additional data alongside the
 *   particle state without having to use a \c thread_local paradigm . (In
 *   other words, this allows us to access the track data on a different CPU
 *   thread from the one actually performing the tracking.)
 *
 * The two key methods for gathering data from Celeritas are \c accum_counters
 * which updates counters that lives in host memory (regardless of whether the
 * main particle data is on device), and the \c StepDiagnosticExecutor class
 * which updates on-device data from the particle states. The latter class is
 * instantiated and run in what is effectively a "parallel for" using a \c
 * ActionLauncher class (GPU) or \c launch_action function (CPU).
 */
class StepDiagnostic final : public CoreStepActionInterface,
                             public AuxParamsInterface,
                             public ParamsDataInterface<StepParamsData>
{
  public:
    // Construct and add to core params
    static std::shared_ptr<StepDiagnostic>
    make_and_insert(CoreParams const& core);

    // Construct with IDs
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

    //! Index of this class instance in the action registry
    ActionId action_id() const final { return sad_.action_id(); }
    //! Ordering of the action inside the step
    StepActionOrder order() const final { return StepActionOrder::user_post; }
    // Execute the action with host data
    void step(CoreParams const& params, CoreStateHost& state) const final;
    // Execute the action with device data
    void step(CoreParams const& params, CoreStateDevice& state) const final;
    //!@}

    //!@{
    //! \name Aux params interface

    //! Index of this class instance in the aux registry
    AuxId aux_id() const final { return aux_id_; }
    // Build state data for a stream
    UPState create_state(MemSpace m, StreamId id, size_type size) const final;
    //!@}

    //!@{
    //! \name Data interface

    //! Access shared setup data on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }
    //! Access shared setup data on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }
    //!@}

  private:
    //// DATA ////

    StaticActionData sad_;
    AuxId aux_id_;
    CollectionMirror<StepParamsData> mirror_;

    //// HELPER FUNCTIONS ////
    void accum_counters(CoreStateCounters const& counters,
                        HostStepStatistics& stats) const;
};

//---------------------------------------------------------------------------//
}  // namespace example
}  // namespace celeritas
