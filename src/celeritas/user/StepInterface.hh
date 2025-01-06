//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>

#include "corecel/Types.hh"

#include "StepData.hh"  // IWYU pragma: export

namespace celeritas
{
//---------------------------------------------------------------------------//
// FIXME: rename this to something less confusing
template<MemSpace M>
struct StepState
{
    // TODO: step state params
    //! Host pointer to externally owned step state data reference
    StepStateData<Ownership::reference, M> const& steps;
    // TODO: aux state vec
    //! Stream ID (local to each core state)
    StreamId stream_id;
};

//---------------------------------------------------------------------------//
/*!
 * Callback class to gather and process data from many tracks at a single step.
 *
 * The filtering mechanism allows different step interfaces to gather data from
 * different detector volumes. Filtered step interfaces cannot be combined with
 * unfiltered in a single hit collector. (FIXME: maybe we need a slightly
 * different class hierarchy for the two cases?) If detectors are in use, and
 * all \c StepInterface instances in use by a \c StepCollector select the
 * "nonzero_energy_deposition" flag, then the \c StepStateData::detector entry
 * for a thread with no energy deposition will be cleared even if it is in a
 * sensitive detector. Otherwise entries with zero energy deposition will
 * remain.
 */
class StepInterface
{
  public:
    //@{
    //! \name Type aliases
    using HostStepState = StepState<MemSpace::host>;
    using DeviceStepState = StepState<MemSpace::device>;
    using MapVolumeDetector = std::map<VolumeId, DetectorId>;
    //@}

    //! Filtering to apply to the gathered data for this step.
    struct Filters
    {
        //! Only select data from these volume IDs and map to detectors
        MapVolumeDetector detectors;
        //! Only select data with nonzero energy deposition (if detectors)
        bool nonzero_energy_deposition{false};
    };

  public:
    //! Detector filtering required for this interface
    virtual Filters filters() const = 0;

    //! Selection of data required for this interface
    virtual StepSelection selection() const = 0;

    //! Process CPU-generated hit data
    virtual void process_steps(HostStepState) = 0;

    //! Process device-generated hit data
    virtual void process_steps(DeviceStepState) = 0;

    // TODO: hook for end-of-event and/or end-of-run

  protected:
    // Protected destructor prevents deletion of pointer-to-interface
    ~StepInterface() = default;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
