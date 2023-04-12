//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/KNDemoIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include <nlohmann/json.hpp>

#include "corecel/Types.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/grid/UniformGridData.hh"

#include "KNDemoKernel.hh"

namespace demo_interactor
{
//---------------------------------------------------------------------------//
// Classes
//---------------------------------------------------------------------------//
//! Input for a single run
struct KNDemoRunArgs
{
    using size_type = celeritas::size_type;
    using GridParams = celeritas::UniformGridData;

    double energy;
    unsigned int seed;
    size_type num_tracks;
    size_type max_steps;
    GridParams tally_grid;
};

//! Output from a single run
struct KNDemoResult
{
    using size_type = celeritas::size_type;

    std::vector<double> time;  //!< Real time per step
    std::vector<size_type> alive;  //!< Num living tracks per step
    std::vector<double> edep;  //!< Energy deposition along the grid
    double total_time = 0;  //!< All time
};

//---------------------------------------------------------------------------//
// JSON I/O functions
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, DeviceGridParams const& value);
void from_json(nlohmann::json const& j, DeviceGridParams& value);

void to_json(nlohmann::json& j, KNDemoRunArgs const& value);
void from_json(nlohmann::json const& j, KNDemoRunArgs& value);

void to_json(nlohmann::json& j, KNDemoResult const& value);
void from_json(nlohmann::json const& j, KNDemoResult& value);

//---------------------------------------------------------------------------//
}  // namespace demo_interactor

namespace celeritas
{
void to_json(nlohmann::json& j, UniformGridData const& value);
void from_json(nlohmann::json const& j, UniformGridData& value);
}  // namespace celeritas
