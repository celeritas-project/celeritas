//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Physics.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Events.hh"
#include "MucfPhysics.hh"
#include "OpticalPhysics.hh"
#include "PhysicsProcess.hh"
#include "ProcessBuilder.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Electromagnetic physics processes and options.
 *
 * \todo The ProcessBuilder is the "general" process builder type and should be
 * refactored once import data is moved into the `inp` classes. The \c
 * user_processes can be set externally or via
 * \c FrameworkInput.geant.ignore_processes.
 */
struct EmPhysics
{
    //! Bremsstrahlung process
    BremsstrahlungProcess brems;
    //! Electron+positron pair production process
    PairProductionProcess pair_production;
    //! Photoelectric effect
    PhotoelectricProcess photoelectric;

    //! Atomic relaxation
    AtomicRelaxation atomic_relaxation;

    //!@{
    //! \name Energy loss and slowing down

    // TODO: currently eloss fluctuations are set up via geant importer, then
    // read into ImportEmParams
#if 0
     //! Energy loss fluctuations
     bool eloss_fluct{true};
#endif
    //
    //!@}

    //! Add custom user processes
    ProcessBuilderMap user_processes;
};

//---------------------------------------------------------------------------//
/*!
 * Optical physics processes, options, and surface definitions.
 *
 * If scintillation or Cherenkov is enabled, optical photons will be generated.
 */
struct OpticalPhysics
{
    //! Optionally generate photons from EM processes
    OpticalGenPhysics gen;
    //! Optical photons interact inside materials
    OpticalBulkPhysics bulk;
    //! Optical photons interact with material boundaries
    OpticalSurfacePhysics surfaces;

    //! Whether optical physics is enabled
    explicit operator bool() const { return gen || bulk || surfaces; }
};

//---------------------------------------------------------------------------//
/*!
 * Set up physics options.
 *
 * \todo Move optical and hadronic physics options from
 *       \c celeritas::GeantPhysicsOptions
 * \todo Move particle data from \c celeritas::ImportParticle
 * \todo Add function for injecting user processes for
 *       \c celeritas::PhysicsParams
 */
struct Physics
{
    //! Electromagnetic physics processes
    EmPhysics em;

    //! Muon-catalyzed fusion physics
    MucfPhysics mucf;

    //! Physics for optical photons
    OpticalPhysics optical;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
