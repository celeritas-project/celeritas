//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ExampleStepCollector.cc
//---------------------------------------------------------------------------//
#include "ExampleStepCollector.hh"

#include <algorithm>
#include <tuple>

#include "corecel/cont/Range.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
void copy_array(double dst[3], const Real3& src)
{
    std::copy(src.begin(), src.end(), dst);
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
void ExampleStepCollector::operator()(StateHostRef const& data)
{
    for (auto tid : range(ThreadId{data.size()}))
    {
        TrackId track = data.track[tid];
        if (!track)
        {
            // Skip inactive slot
            continue;
        }

        Step new_step;
        new_step.event = data.event[tid].get();
        new_step.track = track.unchecked_get();
        new_step.step  = data.track_step_count[tid];

        const auto& pre_step = data.points[StepPoint::pre];
        new_step.volume      = pre_step.volume[tid].get();
        copy_array(new_step.pos, pre_step.pos[tid]);
        copy_array(new_step.dir, pre_step.dir[tid]);

        steps_.push_back(new_step);
    }
}

//---------------------------------------------------------------------------//
void ExampleStepCollector::sort()
{
    std::sort(
        steps_.begin(), steps_.end(), [](const Step& lhs, const Step& rhs) {
            return std::make_tuple(lhs.event, lhs.track, lhs.step)
                   < std::make_tuple(rhs.event, rhs.track, rhs.step);
        });
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
