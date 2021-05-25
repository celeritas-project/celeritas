//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include <nlohmann/json.hpp>
#include "base/Types.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Input for a single run.
 */
struct LDemoArgs
{
    using size_type = celeritas::size_type;

    // Problem definition
    std::string geometry_filename; //!< Path to GDML file
    std::string physics_filename;  //!< Path to ROOT exported Geant4 data
    std::string hepmc3_filename;   //!< Path to hepmc3 event data

    // Source definition (TODO: specify particle type, distribution, ...?)
    // double energy{}; //!< use hepmc3 input?

    // Control
    unsigned int seed{};
    size_type    max_steps{};

    //! Whether the run arguments are valid
    explicit operator bool() const
    {
        return !geometry_filename.empty() && !physics_filename.empty()
               && !hepmc3_filename.empty() && max_steps > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Tallied result and timing from run.
 */
struct LDemoResult
{
    using size_type = celeritas::size_type;

    std::vector<double>    time;  //!< Real time per step
    std::vector<size_type> alive; //!< Num living tracks per step
    std::vector<double>    edep;  //!< Energy deposition along the grid
    double                 total_time = 0; //!< All time
};

void to_json(nlohmann::json& j, const LDemoArgs& value);
void from_json(const nlohmann::json& j, LDemoArgs& value);

void to_json(nlohmann::json& j, const LDemoResult& value);

//---------------------------------------------------------------------------//
} // namespace demo_loop
