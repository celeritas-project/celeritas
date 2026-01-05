//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Problem.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/inp/Model.hh"

#include "Control.hh"
#include "Diagnostics.hh"
#include "Field.hh"
#include "Physics.hh"
#include "Scoring.hh"
#include "Tracking.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Configure Celeritas input.
 *
 * Ideally, these input structs match up with the runtime classes that
 * they help construct. Future restructuring of the code will result in more
 * direct correspondence.
 *
 * Currently, the problem is supplemented by (an older implementation) \c
 * ImportData which contains much of the heavy-weight physics data. All of
 * Celeritas (except unit tests, for now) problems are initialized through the
 * \c celeritas::inp::StandaloneInput and \c celeritas::inp::FrameworkInput
 * (which for now is constructed via \c celeritas::SetupOptions in \c accel).
 * Those specify system configurations and how to load the physics/problem.
 *
 * The \c Problem specifies what and how Celeritas will run. All of Celeritas'
 * user-facing capabilities should be represented in this data structure or its
 * members, \em and where appropriate it will offer extension points for
 * integrating user physics and diagnostics.
 */
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Celeritas problem input definition.
 *
 * This should specify all the information necessary to track particles
 * within Celeritas for offloading or standalone execution. (It does \em not
 * contain system configuration such as GPU, or event/offload information.)
 *
 * Multiple problems can be run independently across the same program
 * execution.
 *
 * Eventually this class and its daughters will subsume all the data in
 * \c celeritas/io/ and all the input options from Models, Processes,
 * Params, and other classes that are not implementation details.
 *
 * After loading, the struct will be able to be serialized to ROOT or JSON or
 * some other struct for reproducibility.
 */
struct Problem
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

    //! Low-level performance tuning and simulation control options
    Control control;
    //! Monte Carlo tracking, performance, and debugging diagnostics
    Diagnostics diagnostics;
};

//---------------------------------------------------------------------------//
/*!
 * Celeritas optical-only problem input definition.
 */
struct OpticalProblem
{
    //! Geometry, material, and region definitions
    Model model;
    //! Physics models and options
    OpticalPhysics physics;
    //! Optical photon generation mechanism
    OpticalGenerator generator;
    //! Hard cutoffs for counters
    OpticalTrackingLimits limits;
    //! Per-process state sizes for optical tracking loop
    OpticalStateCapacity capacity;
    //! Number of streams
    size_type num_streams{};
    //! Random number generator seed
    unsigned int seed{};
    //! Set up step or action timers
    Timers timers;
    //! Write Celeritas diagnostics to this file ("-" is stdout)
    std::string output_file{"-"};
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
