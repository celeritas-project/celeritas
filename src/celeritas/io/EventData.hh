//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <array>
#include <map>
#include <vector>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Example of a calorimeter hit.
 */
struct EventHitData
{
    int volume{0};  //!< Logical volume ID
    int copy_num{0};  //!< Physical volume copy number
    double energy_dep{0};  //!< Energy deposition [MeV]
    double time{0};  //!< Pre-step global time [s]
};

//---------------------------------------------------------------------------//
/*!
 * Pre- and post-step point information.
 */
struct EventStepPointData
{
    double energy{0};  //!< Kinetic energy [MeV]
    std::array<double, 3> dir{0, 0, 0};  //!< Unit vector
    std::array<double, 3> pos{0, 0, 0};  //!< [cm]
};

//---------------------------------------------------------------------------//
/*!
 * Particle step data. Arrays are for pre- and post-steps.
 *
 * TODO: Map and add a unified process/action id; Add PDG, add track and parent
 * IDs.
 */
struct EventStepData
{
    EventHitData hit;
    double length{0};  //!< Step length [cm]
    std::array<EventStepPointData, 2> step_points;  //!< Pre- and post-steps
};

//---------------------------------------------------------------------------//
/*!
 * Event data to be used within a Geant4/Celeritas offloading application.
 *
 * The steps are designed to be assigned to each sensitive volume, so that a
 * vector of steps of a given volume can be retrieved by doing
 *
 * \code
 * auto const& sd_steps = event_data.steps[sensdet_id];
 * for (auto const& step : sd_steps)
 * {
 *     // Access step information from a given detector in this event.
 * }
 * \endcode
 *
 * Therefore, sensitive detector IDs must be contiguously assigned and mapped
 * to their sensitive detector name at startup.
 */
struct EventData
{
    int event_id{0};
    std::vector<std::vector<EventStepData>> steps;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
