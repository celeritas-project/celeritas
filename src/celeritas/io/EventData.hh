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
 * Particle step data. Arrays are for pre- and post-steps.
 *
 * TODO: Map and add a unified process/action id
 */
struct EventStepData
{
    int volume{0};  //!< Logical volume ID
    int pdg{0};
    int parent_id{0};
    int track_id{0};
    double energy_loss{0};  //!< Same as energy deposition [MeV]
    double length{0};  //!< Step length [cm]

    // Pre- and post-step information
    double energy[2]{0};  //!< Kinetic energy [MeV]
    std::array<double, 3> dir[2]{{0, 0, 0}};  //!< Unit vector
    std::array<double, 3> pos[2]{{0, 0, 0}};  //!< [cm]
    double time[2]{0};  //!< Global coordinate time [s]
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
 *     // Access step from a given sensitive detector in this event.
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
