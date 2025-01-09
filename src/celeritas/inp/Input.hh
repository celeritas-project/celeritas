//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Input.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Diagnostics.hh"
#include "Field.hh"
#include "Model.hh"
#include "Physics.hh"
#include "Scoring.hh"
#include "Tracking.hh"
#include "Tuning.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Celeritas setup.
 *
 * OPEN QUESTIONS:
 * - Some parameters we want to use \em only when we're driving Geant4 and
 *   aren't used for Celeritas: hadronic physics list, certain physics options.
 *   Do those go here or in a separate data structure?
 * - Do we add callbacks to the "physics" section to inject new processes? To
 *   the "scoring"/"diagnostics" to add additional actions?
 */
struct Input
{
    //! Geometry, material, and region definitions
    Model model;
    //! Physics models and options
    Physics physics;
    //! Set up the magnetic field
    Field field;
    //! Manage scoring of hits and other quantities
    Scoring scoring;

    //! Tuning options that affect the physics
    Tracking tracking;
    //! Low-level performance tuning options
    Tuning tuning;

    //! Monte Carlo tracking, performance, and debugging diagnostics
    Diagnostics diagnostics;
};

void framework_adjust_options(Input& inp)
{
    inp.scoring.sd->energy_deposition = true;
    inp.tuning.device = my_framework_option.use_gpu;

    if (my_framework_option.field_enabled)
    {
        UniformField field;
        field.strength = {1, 2, 3};
        inp.field = std::move(field);
    }

    inp.physics.em.rayleigh = {};
}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
