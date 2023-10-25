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
struct StepData
{
    int volume{0};  //!< Logical volume ID
    double energy_loss{0};  //!< Same as energy deposition in hit data
    double length{0};  //!< Step length

    // Pre- and post-step information
    double energy[2]{0};  //!< Kinetic energy
    std::array<double, 3> dir[2]{{0, 0, 0}};  //!< Unit vector
    std::array<double, 3> pos[2]{{0, 0, 0}};  //!< [cm]
    double time[2]{0};  //!< Global coordinate time
};

//---------------------------------------------------------------------------//
/*!
 * Basic sensitive hit data.
 */
struct HitData
{
    unsigned int id{0};  //!< Volume's copy number
    double edep{0};  //!< Energy deposition
    double time{0};  //!< Global coordinate time
};

//---------------------------------------------------------------------------//
/*!
 * Event data to be used within a Geant4/Celeritas offloading application.
 *
 * VolumeId is a contiguous Id available globally and mapped to its
 * respective volume name.
 */
struct EventData
{
    using SensDetId = int;

    int event_id{0};
    std::vector<StepData> steps;
    std::map<SensDetId, std::vector<HitData>> hits;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
