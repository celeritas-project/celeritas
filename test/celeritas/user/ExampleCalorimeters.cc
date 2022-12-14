//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ExampleCalorimeters.cc
//---------------------------------------------------------------------------//
#include "ExampleCalorimeters.hh"

#include "celeritas/geo/GeoParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Convert calorimeter volume names to volume IDs.
 */
ExampleCalorimeters::ExampleCalorimeters(const GeoParams&                geo,
                                         const std::vector<std::string>& volumes)
{
    CELER_EXPECT(!volumes.empty());

    for (const auto& name : volumes)
    {
        auto vid = geo.find_volume(name);
        CELER_VALIDATE(vid, << "volume '" << name << "' does not exist");
        detectors_.push_back(vid);
    }

    this->clear();
    CELER_ENSURE(detectors_.size() == deposition_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Map volume names to detector IDs and exclude tracks with no deposition.
 */
auto ExampleCalorimeters::filters() const -> Filters
{
    Filters result;

    for (auto didx : range<DetectorId::size_type>(detectors_.size()))
    {
        result.detectors[detectors_[didx]] = DetectorId{didx};
    }

    result.nonzero_energy_deposition = true;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Gather event IDs and local energy deposition.
 */
StepSelection ExampleCalorimeters::selection() const
{
    StepSelection result;
    result.event_id          = true;
    result.energy_deposition = true;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies.
 */
void ExampleCalorimeters::execute(StateHostRef const& data)
{
    for (auto tid : range(ThreadId{data.size()}))
    {
        DetectorId det = data.detector[tid];
        if (!det)
        {
            // Skip thread slot that's inactive or not in a calorimeter
            continue;
        }

        if (!event_)
        {
            event_ = data.event_id[tid];
        }
        // Only tally results from one event at a time
        CELER_ASSERT(event_ == data.event_id[tid]);

        CELER_ASSERT(det < deposition_.size());
        real_type edep
            = value_as<units::MevEnergy>(data.energy_deposition[tid]);
        CELER_ASSERT(edep > 0);
        deposition_[det.unchecked_get()] += edep;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Reset for a new event.
 */
void ExampleCalorimeters::clear()
{
    event_ = {};
    deposition_.assign(detectors_.size(), 0);
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
