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
