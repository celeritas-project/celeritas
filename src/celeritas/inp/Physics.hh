//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Physics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/io/Label.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/AtomicNumber.hh"

#include "PhysicsProcess.hh"
#include "ProcessBuilder.hh"
#include "SurfacePhysics.hh"

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
 * \todo Move cherenkov/scintillation to a OpticalGenPhysics class.
 */
struct OpticalPhysics
{
    //!@{
    /*! \name Optical photon generation from EM particles
     *
     *  \todo Replace with physics input data
     */

    //! Generate Cherenkov photons
    bool cherenkov{false};

    //! Generate scintillation photons
    bool scintillation{false};
    //!@}

    //!@{
    //! \name Optical surface physics and properties
    SurfacePhysics surfaces;
    //!@}

    //! \todo Move optical bulk models here

    //! Whether optical physics is enabled
    explicit operator bool() const
    {
        return cherenkov || scintillation || surfaces;
    }
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
 *
 * \todo How to better group these, especially when adding
 * hadronic/photonuclear/decay/...?
 */
struct Physics
{
    //! Physics that applies to offloaded EM particles
    EmPhysics em;

    //! Physics for optical photons
    OpticalPhysics optical;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
