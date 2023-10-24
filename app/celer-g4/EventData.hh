//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/EventData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <array>
#include <map>

// #include "celeritas/io/ImportProcess.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Map detector id and name separately to be stored on the host. Use only
 * detector ID to map hits on both host and device.
 */
using DetectorIdMap = std::map<unsigned int, std::string>;

//---------------------------------------------------------------------------//
/*!
 * Sensitive hit data.
 */
struct HitData
{
    unsigned int id{0};  //!< Detector ID
    double edep{0};  //!< Energy deposition
    double time{0};  //!< Time (global coordinate)
};

//---------------------------------------------------------------------------//
/*!
 * Particle step data.
 */
struct StepData
{
    enum StepType
    {
        pre,
        post,
        size_
    };

    // ImportProcessClass action_id{ImportProcessClass::size_};
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
 * Hit collection of an event.
 */
struct EventData
{
    using DetectorId = int;

    int event_id{0};
    std::vector<StepData> steps;
    std::map<DetectorId, std::vector<HitData>> hits;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
