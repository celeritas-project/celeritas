//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
//! Physics list selection
enum class PhysicsList
{
    ftfp_bert,
    geant_physics_list,
    size_,
};

//---------------------------------------------------------------------------//
/*!
 * Input for a single run.
 */
struct RunInput
{
    // Path to HepMC3 event record file
    std::string event_file;

    // Setup options for generating primaries programmatically
    PrimaryGeneratorOptions primary_options;

    // Physics setup options
    PhysicsList physics_list{PhysicsList::ftfp_bert};
    GeantPhysicsOptions physics_options;

    // Field driver options
    FieldDriverOptions field_options;

    // Diagnostics
    bool step_diagnostic{false};
    int step_diagnostic_bins{1000};

    //! Whether the run arguments are valid
    explicit operator bool() const
    {
        return (primary_options || !event_file.empty())
               && (step_diagnostic_bins > 0 || !step_diagnostic);
    }
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
