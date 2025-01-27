//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Problem.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Control.hh"
#include "Diagnostics.hh"
#include "Field.hh"
#include "Model.hh"
#include "Physics.hh"
#include "Scoring.hh"
#include "Tracking.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Configure Celeritas input.
 *
 * Note that many of these input structs match up with the runtime classes that
 * they help construct. Future restructuring of the code may result in more
 * direct correspondence.
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
}  // namespace inp
}  // namespace celeritas
