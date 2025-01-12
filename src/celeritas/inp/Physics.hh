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
#include "celeritas/Types.hh"
#include "celeritas/phys/AtomicNumber.hh"

#include "PhysicsProcess.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Electromagnetic physics processes and options.
 */
struct EmPhysics
{
    std::optional<BremsProcess> brems{std::in_place};
#if 0
    // TODO
    std::optional<ComptonScatProcess> comptonscat{std::in_place};
    std::optional<CoulombScatProcess> coulombscat{std::in_place};
    std::optional<IoniProcess> ioni{std::in_place};
    std::optional<AnniProcess> anni{std::in_place};
    std::optional<ConversionProcess> conversion{std::in_place};
    std::optional<PhotoelectricProcess> photoelectric{std::in_place};
    std::optional<RayleighScatProcess> rayleighscat{std::in_place};
#endif

    //!@{
    //! \name Energy loss and slowing down
    //! Second-order spline interpolation for energy loss
    bool eloss_spline{false};
#if 0
     //! Energy loss fluctuations
     bool eloss_fluct{true};
#endif
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Optical physics processes and options.
 */
struct OpticalPhysics
{
};

//---------------------------------------------------------------------------//
/*!
 * Hadronic physics processes and options.
 *
 * This can be used to enable or set up Geant4 hadronic physics.
 */
struct HadronicPhysics
{
};

//---------------------------------------------------------------------------//
/*!
 * Set up physics options.
 *
 * \todo Move optical and hadronic physics options from \c GeantPhysicsOptions
 * \todo Particle data
 * \todo Function for injecting user processes
 */
struct Physics
{
    std::optional<EmPhysics> em{std::in_place};
    std::optional<OpticalPhysics> optical;
    std::optional<HadronicPhysics> hadronic;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
