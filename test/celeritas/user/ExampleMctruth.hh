//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ExampleMctruth.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"
#include "celeritas/Types.hh"
#include "celeritas/user/StepInterface.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Store all step data in an AOS.
 *
 * Construct a StepCollector with this callback interface to tally data during
 * execution. At the end of the run, for testing, call \c sort to reorder the
 * data by event/track/step, and then access the data with \c steps .
 */
class ExampleMctruth final : public StepInterface
{
  public:
    struct Step
    {
        int event;
        int track;
        int step;
        int volume;  // Beginning of step
        double pos[3];  // Beginning of step
        double dir[3];  // Beginning of step
    };

  public:
    //! Selection of data required for this interface
    Filters filters() const final { return {}; }

    // Return flags corresponding to the "Step" above
    StepSelection selection() const final;

    // Tally host data for a step iteration
    void process_steps(HostStepState) final;

    // Tally device data for a step iteration
    void process_steps(DeviceStepState) final
    {
        CELER_NOT_IMPLEMENTED("device example");
    }

    // Sort tallied tracks by {event, track, step}
    void sort();

    //! Access all steps
    Span<Step const> steps() const { return make_span(steps_); }

    //! Reset after output or whatever
    void clear() { steps_.clear(); }

  private:
    std::vector<Step> steps_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
