//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Physics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "corecel/Types.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
struct AlongStepFactoryInput;

namespace inp
{
#if 0
//---------------------------------------------------------------------------//
/*!
 * Set up multiple scattering options.
 *
 * TODO: some of these are moved from \c ImportEmParameters : we should import
 * more and change them around a bit.
 */
struct MscOptions
{
    MscStepLimitAlgorithm step_limit{MscStepLimitAlgorithm::safety};

    //! MSC range factor for e-/e+
    real_type range_factor{0.04};
    //! MSC safety factor
    real_type safety_factor{0.6};
    //! MSC lambda limit [length]
    real_type lambda_limit{1 * units::millimeter};
    //! Polar angle limit between single and multiple Coulomb scattering
    real_type theta_limit{constants::pi};

    // TODO: unit system support?
    static inline constexpr UnitSystem units{UnitSystem::native};
};
#endif

//---------------------------------------------------------------------------//
/*!
 * Set up electromagnetic physics options.
 */
struct EmPhysicsOptions
{
    //!@{
    //! \name Energy loss and slowing down
    //! Second-order spline interpolation for energy loss
    bool eloss_spline{false};
    // TODO: //! Energy loss fluctuations
    // TODO: bool eloss_fluct{true};
    // TODO: //! Integral cross section rejection
    // TODO: bool integral_xs_rejection{true};
    //!@}

    //!@{
    //! \name Model options
    //! Use LPM corrections for high-energy bremsstrahlung and pair production
    bool lpm{true};
    //! Use combined SB/relativistic interactor for bremsstrahlung
    bool brem_combined{false};
    //!@}

    //! Hardcoded maximum step for charged particles (none if zero)
    real_type step_limit{};
};

//---------------------------------------------------------------------------//
/*!
 * Set up physics options.
 *
 * TODO: refactor ignore_processes so it ties in the with IO classes.
 */
struct Physics
{
    //! Electromagnetic physics options
    EmPhysicsOptions em_options;

    //! Do not use Celeritas physics for the given Geant4 process names
    std::vector<std::string> ignore_processes;

    //! Import physics from a file instead of Geant4
    std::string physics_file;

    // TODO: particle selection
    // TODO: user process builder
    // TODO: where/how should celer-g4 set up physics list?
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
