//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ExampleCalorimeters.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "celeritas/geo/GeoParamsFwd.hh"
#include "celeritas/user/StepInterface.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct a "detector" that accumulates energy deposition at every step.
 *
 * This is not the most efficient way to integrate data over multiple steps
 * (especially on device) but it is the most user-friendly way to do it.
 *
 * The current implementation only works with a single event at a time.
 */
class ExampleCalorimeters final : public StepInterface
{
  public:
    // Construct from detectors to monitor
    ExampleCalorimeters(GeoParams const& geo,
                        std::vector<std::string> const& volumes);

    // Filter data being gathered
    Filters filters() const final;

    // Return flags corresponding to the "Step" above
    StepSelection selection() const final;

    // Tally host data for a step iteration
    void process_steps(HostWTFStepState) final;

    // Tally device data for a step iteration
    void process_steps(DeviceWTFStepState) final
    {
        CELER_NOT_IMPLEMENTED("device example");
    }

    //! Access summed energy deposition [MeV] for all volumes
    Span<real_type const> deposition() const { return make_span(deposition_); }

    // Reset for a new event
    void clear();

  private:
    std::vector<VolumeId> detectors_;
    std::vector<real_type> deposition_;
    EventId event_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
