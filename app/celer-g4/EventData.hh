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
    std::array<double, 3> pos{0, 0, 0};  //!< Position (global coordinate)
};

#if 0
//---------------------------------------------------------------------------//
/*!
 * Particle step data.
 */
struct StepData
{
    enum class StepType
    {
        pre,
        post,
        size_
    };

    StepType step_type{StepType::size_};
    ImportProcessClass action_id{ImportProcessClass::size_};
    unsigned int detector_id;  //!< Defined in SD Manager
    double kinetic_energy{0};  //!< [MeV]
    double energy_loss{0};  //!< [MeV]
    double length{0};  //!< [cm]
    std::array<double, 3> direction{0, 0, 0};  //!< Unit vector
    std::array<double, 3> position{0, 0, 0};  //!< [cm]
    double global_time{0};  //!< [s]
};
#endif

//---------------------------------------------------------------------------//
/*!
 * Hit collection of an event.
 */
struct EventData
{
    using TrackId = int;
    using DetectorId = int;

    int event_id{0};
    // std::vector<StepData> steps;
    std::map<DetectorId, std::vector<HitData>> hits;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
