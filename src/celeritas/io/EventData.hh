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
 * Particle pre- and post-step data.
 */
struct StepData
{
    enum StepType
    {
        pre,
        post,
        size_
    };

    // TODO: add process/action id
    unsigned int detector_id[2]{0};  //!< Defined in SD Manager
    double energy[2]{0};  //!< [MeV]
    double energy_loss{0};  //!< [MeV]
    double length{0};  //!< [cm]
    std::array<double, 3> dir[2]{{0, 0, 0}};  //!< Unit vector
    std::array<double, 3> pos[2]{{0, 0, 0}};  //!< [cm]
    double time[2]{0};  //!< [s]
};

//---------------------------------------------------------------------------//
/*!
 * Basic sensitive hit data.
 */
struct HitData
{
    unsigned int id{0};  //!< Detector ID
    double edep{0};  //!< Energy deposition
    double time{0};  //!< Time (global coordinate)
};

//---------------------------------------------------------------------------//
/*!
 * Event data to be used within a Geant4/Celeritas offloading application.
 *
 * DetectorId is a contiguous Id available globally and mapped to its
 * respective volume name.
 */
struct EventData
{
    using DetectorId = int;

    int event_id{0};
    std::vector<StepData> steps;
    std::map<DetectorId, std::vector<HitData>> hits;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
