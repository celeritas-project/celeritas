//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/MakeCelerOptions.cc
//---------------------------------------------------------------------------//
#include "MakeCelerOptions.hh"

#include <accel/AlongStepFactory.hh>
#include <accel/SetupOptions.hh>

//---------------------------------------------------------------------------//
/*!
 * Build options to set up Celeritas.
 */
celeritas::SetupOptions MakeCelerOptions()
{
    celeritas::SetupOptions opts;

    opts.max_num_tracks = 1024 * 16;
    opts.initializer_capacity = 1024 * 128 * 4;
    opts.secondary_stack_factor = 2.0;
    opts.ignore_processes = {"CoulombScat"};

    // Set along-step factory with zero field
    opts.make_along_step = celeritas::UniformAlongStepFactory();

    // Save diagnostic information
    opts.output_file = "celeritas-offload-diagnostic.json";

    return opts;
}
