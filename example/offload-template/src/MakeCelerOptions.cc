//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/MakeCelerOptions.cc
//---------------------------------------------------------------------------//
#include "MakeCelerOptions.hh"

#include <accel/AlongStepFactory.hh>
#include <accel/SetupOptions.hh>

#include "EventAction.hh"
#include "StepDiagnostic.hh"

namespace celeritas
{
namespace example
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Add the step diagnostic.
 */
void AddUserActions(CoreParams const& params)
{
    // Add the diagnostic to the stepping loop
    auto diagnostic = StepDiagnostic::make_and_insert(params);
    CELER_ASSERT(diagnostic);

    // Save it to transfer and print at the end of every event
    EventAction::SetStepDiagnostic(std::move(diagnostic));
}
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Build options to set up Celeritas.
 */
celeritas::SetupOptions MakeCelerOptions()
{
    celeritas::SetupOptions opts;

    // NOTE: these can be omitted if using a .mac input file
    opts.max_num_tracks = 1024 * 16;
    opts.initializer_capacity = 1024 * 128 * 4;

    opts.ignore_processes = {"CoulombScat"};

    // Set along-step factory with zero field
    opts.make_along_step = celeritas::UniformAlongStepFactory();

    // Add a callback to adding user actions
    opts.add_user_actions = AddUserActions;

    return opts;
}

}  // namespace example
}  // namespace celeritas
