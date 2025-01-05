//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Physics.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Field.h"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Set up multiple scattering options.
 */
struct MscOptions
{
    MscStepLimitAlgorithm step_limit{MscStepLimitAlgorithm::safety};

    //! MSC range factor for e-/e+
    double range_factor{0.04};
    //! MSC safety factor
    double safety_factor{0.6};
    //! MSC lambda limit [length]
    double lambda_limit{1 * units::millimeter};
    //! Polar angle limit between single and multiple Coulomb scattering
    double theta_limit{constants::pi};

    // TODO: unit system support?
    static inline constexpr UnitSystem units{UnitSystem::native};
};

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
    // TODO: //! LPM corrections for bremsstrahlung and pair production
    // TODO: bool lpm{true};
    //! Use combined SB/relativistic interactor
    bool brem_combined{false};
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Set up physics options.
 *
 * TODO: refactor ignore_processes so it ties in the with IO classes.
 */
struct Physics
{
    using AlongStepFactory
        = std::function<SPConstAction(AlongStepFactoryInput const&)>;

    //! Electromagnetic physics options
    EmPhysicsOptions em_options;

    //! Do not use Celeritas physics for the given Geant4 process names
    std::vector<std::string> ignore_processes;

    //! REMOVE: along-step factory
    AlongStepFactory make_along_step;

    // TODO: particle selection
    // TODO: user process builder
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
