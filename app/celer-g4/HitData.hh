//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/HitData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <array>
#include <map>

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
 * Example sensitive hit data.
 */
struct HitData
{
    unsigned int id{0};  //!< Detector id
    double edep{0};  //!< Energy deposition
    double time{0};  //!< Time (global coordinate)
    std::array<double, 3> pos{0, 0, 0};  //!< Position (global coordinate)
};

//---------------------------------------------------------------------------//
/*!
 * Example hit collection of an event.
 */
struct HitEventData
{
    int event_id{0};
    std::map<std::string, std::vector<HitData>> hits;
    // TODO: replace name by id
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
