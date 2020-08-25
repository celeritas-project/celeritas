//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KNDemoIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "KNDemoKernel.hh"
#include "base/Types.hh"
#include <nlohmann/json.hpp>

namespace demo_interactor
{
//---------------------------------------------------------------------------//
// Classes
//---------------------------------------------------------------------------//
//! Input for a single run
struct KNDemoRunArgs
{
    using size_type = celeritas::size_type;

    double        energy;
    unsigned long seed;
    size_type     num_tracks;
    size_type     max_steps;
};

//! Output from a single run
struct KNDemoResult
{
    using size_type = celeritas::size_type;

    std::vector<double>    time;           //!< Real time per step
    std::vector<double>    edep;           //!< Energy deposition per step
    std::vector<size_type> alive;          //!< Num living tracks
    double                 total_time = 0; //!< All time
};

//---------------------------------------------------------------------------//
// JSON I/O functions
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, const CudaGridParams& value);
void from_json(const nlohmann::json& j, CudaGridParams& value);

void to_json(nlohmann::json& j, const KNDemoRunArgs& value);
void from_json(const nlohmann::json& j, KNDemoRunArgs& value);

void to_json(nlohmann::json& j, const KNDemoResult& value);
void from_json(const nlohmann::json& j, KNDemoResult& value);

//---------------------------------------------------------------------------//
} // namespace demo_interactor
