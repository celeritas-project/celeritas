//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Input.hh
//---------------------------------------------------------------------------//
#pragma once
#include "Diagnostics.hh"
#include "Events.hh"
#include "Field.hh"
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
 * There are three categories of input types:
 * - Exclusive Celeritas inputs: state size, other diagnostic and tuning
 *   parameters. Also some parameters we \em cannot deduce from Geant4: e.g.,
 *   sensitive detector attributes.
 * - Parameters that we want to use to drive Geant4 for celer-g4, or
 *   pull from Geant4 if we're not. This would include geometry
 *   definition (GDML), magnetic field, EM
 *   parameters, active processes, and maybe someday the scoring setup.
 * - Problem setup options that we cannot directly understand from Geant4 but
 *   must be provided directly to lower-level Celeritas objects. In particular,
 *   we need to be able to allow users to add custom processes, magnetic
 *   fields, etc.
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
    //! Path to GDML file, or empty if using Geant4.
    std::string geometry_file;

    //! Physics models and options
    Physics physics;
    //! Set up the magnetic field
    Field field;
    //! Set up event input: offloading, event file, or generator
    Events events;
    //! Manage scoring of hits and other quantities
    Scoring scoring;

    //! Tuning options that affect the physics
    Tracking tracking;
    //! Low-levelp erformance tuning options
    Tuning tuning;

    //! Monte Carlo tracking, performance, and debugging diagnostics
    Diagnostics diagnostics;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
