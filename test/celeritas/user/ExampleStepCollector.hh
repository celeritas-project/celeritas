//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ExampleStepCollector.hh
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
 * Construct a StepCollector with this interface to tally data during
 * execution. At the end of the run, for testing, you probably want to sort the
 * data.
 */
class ExampleStepCollector : public StepInterface
{
  public:
    struct Step
    {
        int    event;
        int    track;
        int    step;
        int    volume; // Beginning of step
        double pos[3]; // Beginning of step
        double dir[3]; // Beginning of step
    };

  public:
    // Tally host data for a step iteration
    void operator()(StateHostRef const&) final;

    // Tally device data for a step iteration
    void operator()(StateDeviceRef const&) final
    {
        CELER_NOT_IMPLEMENTED("device example");
    }

    //! Sort tallied tracks by {event, track, step}
    void sort();

    //! Access all steps
    Span<const Step> steps() const { return make_span(steps_); }

    //! Reset after output or whatever
    void clear() { steps_.clear(); }

  private:
    std::vector<Step> steps_;
};

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
