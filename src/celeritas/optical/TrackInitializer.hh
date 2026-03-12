//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/TrackInitializer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "geocel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Optical photon data used to initialize a photon track state.
 *
 * These are created in one of two ways:
 * - by the Generator classes (e.g., \c celeritas::CherenkovGenerator ) via the
 *   \c OpticalCollector from the main Celeritas EM tracking loop, or from the
 *   \c LocalOpticalGenOffload Geant4 interface class
 * - directly by the user (for testing or direct Geant4 offloading) for use
 *   with the \c DirectGeneratorAction .
 *
 * \internal These are converted to tracks in the
 * \c detail::DirectGeneratorExecutor and \c detail::GeneratorExecutor classes.
 * In the latter, used for scintillation and cherenkov generation, the \c
 * TrackInitializer is ephemeral (created locally, never stored).
 */
struct TrackInitializer
{
    //! Photon energy (MeV for consistency with the main Celeritas loop)
    units::MevEnergy energy;
    //! Position [len]
    Real3 position{0, 0, 0};
    //! Direction
    Real3 direction{0, 0, 0};
    //! Polarization vector
    Real3 polarization{0, 0, 0};
    //! Global lab frame time
    real_type time{};
    //! Geant4 primary ID for MC truth
    PrimaryId primary;
    //! Starting volume
    ImplVolumeId volume{};
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
