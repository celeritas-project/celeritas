//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
    using StateHostRef = HostRef<StepStateData>;
    using StateDeviceRef = DeviceRef<StepStateData>;
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
    //! Selection of data required for this interface
    virtual Filters filters() const = 0;

    //! Selection of data required for this interface
    virtual StepSelection selection() const = 0;

    //! Process CPU-generated hit data
    virtual void execute(StateHostRef const&) = 0;

    //! Process device-generated hit data
    virtual void execute(StateDeviceRef const&) = 0;

    // TODO: hook for end-of-event and/or end-of-run

  protected:
    // Protected destructor prevents deletion of pointer-to-interface
    ~StepInterface() = default;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
