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

#include "Events.hh"
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
 * Muon-catalyzed fusion physics options and data import.
 *
 * Minimum requirements for muon-catalyzed fusion:
 * - Muon energy CDF data, required for sampling the outgoing muCF muon, and
 * - Mean cycle rate data for dd, dt, and tt muonic molecules.
 *
 * Muonic atom transfer and muonic atom spin flip are secondary effects.
 */
struct MucfPhysics
{
    template<class T>
    using Vec = std::vector<T>;

    Grid muon_energy_cdf;  //!< CDF for outgoing muCF muon
    Vec<CycleRateData> cycle_rates;  //!< Mean cycle rates for muonic molecules
    Vec<AtomTransferRateData> atom_transfer;  //!< Muon atom transfer rates
    Vec<AtomSpinFlipRateData> atom_spin_flip;  //!< Muon atom spin flip rates

    //! Whether muon-catalyzed fusion physics is enabled
    explicit operator bool() const
    {
        return muon_energy_cdf && !cycle_rates.empty();
    }

    /*!
     * Construct hardcoded muon-catalyzed fusion physics data.
     *
     * \note
     * This will be replaced by importing the data from output files.
     * The grid data are large, so we may want to have a separate header to
     * store them in the meantime, otherwise this function will be thousands of
     * lines long.
     */
    static MucfPhysics from_default()
    {
        MucfPhysics result;

        //! \todo Initialize hardcoded muon_energy_cdf grid
        //! \todo Initialize hardcoded cycle_rate data for dt, dd, tt

        return result;
    }
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
 * \todo Move \c OpticalGenerator to \c OpticalGenPhysics or elsewhere
 *
 * \todo How to better group these, especially when adding
 * hadronic/photonuclear/decay/...?
 */
struct Physics
{
    //! Physics that applies to offloaded EM particles
    EmPhysics em;

    //! Muon-catalyzed fusion physics
    MucfPhysics mucf;

    //! Physics for optical photons
    OpticalPhysics optical;
    //! Optical photon generation mechanism
    OpticalGenerator optical_generator;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
