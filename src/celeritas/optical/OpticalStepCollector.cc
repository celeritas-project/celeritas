//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalStepCollector.cc
//---------------------------------------------------------------------------//
#include "OpticalStepCollector.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Particle Params data.
 */
OpticalStepCollector::OpticalStepCollector(SPParticleParams particle_params,
                                           unsigned int num_streams)
    : particles_(std::move(particle_params))
{
    CELER_EXPECT(particle_params);
    CELER_EXPECT(num_streams);
    step_data_ = std::make_unique<OpticalStepCollectorData[]>(num_streams);
}

//---------------------------------------------------------------------------//
/*!
 * Collect optical step data for all active threads on the host.
 */
void OpticalStepCollector::process_steps(HostStepState state)
{
    this->process(state);
}

//---------------------------------------------------------------------------//
/*!
 * Collect optical step data for all active threads on device.
 */
void OpticalStepCollector::process_steps(DeviceStepState state)
{
    this->process(state);
}

//---------------------------------------------------------------------------//
/*!
 * Collect optical step data. The collected data is available via
 * \c this->get(tid)
 */
template<class T>
void OpticalStepCollector::process(T state)
{
    for (auto const tid : range(TrackSlotId{state.steps.size()}))
    {
        auto const& ssd = state.steps.data;
        if (!ssd.track_id[tid])
        {
            // Skip inactive tracks
            continue;
        }

        step_data_[tid.unchecked_get()].step_length = ssd.step_length[tid];
        step_data_[tid.unchecked_get()].time
            = ssd.points[StepPoint::pre].time[tid];

        for (auto const sp : range(StepPoint::size_))
        {
            auto const& point = ssd.points[sp];
            step_data_[tid.unchecked_get()].points[sp].pos = point.pos[tid];
            step_data_[tid.unchecked_get()].points[sp].speed
                = this->speed(ssd.particle[tid], point.energy[tid]);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
