//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsOptions.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Particle-dependent physics configuration options.
 *
 * These parameters have different values for light particles
 * (electrons/positrons) and heavy particles (muons/hadrons).
 *
 * Input options are:
 * - \c min_range: below this value, there is no extra transformation from
 *   particle range to step length.
 * - \c max_step_over_range: at higher energy (longer range), gradually
 *   decrease the maximum step length until it's this fraction of the tabulated
 *   range.
 * - \c lowest_energy: tracking cutoff kinetic energy.
 * - \c displaced: whether MSC lateral displacement is enabled for e-/e+
 * - \c range_factor: used in the MSC step limitation algorithm to restrict the
 *   step size to \f$ f_r \cdot max(r, \lambda) \f$ at the start of a track or
 *   after entering a volume, where \f$ f_r \f$ is the range factor, \f$ r \f$
 *   is the range, and \f$ \lambda \f$ is the mean free path.
 * - \c step_limit_algorithm: algorithm used to determine the MSC step limit.
 *
 * NOTE: min_range/max_step_over_range are not accessible through Geant4, and
 * they can also be set to be different for electrons, mu/hadrons, and ions
 * (they are set in Geant4 with \c G4EmParameters::SetStepFunction()).
 */
struct ParticleOptions
{
    using Energy = units::MevEnergy;

    //!@{
    //! \name Range calculation
    real_type min_range;
    real_type max_step_over_range;
    //!@}

    //!@{
    //! \name Energy loss
    Energy lowest_energy;
    //!@}

    //!@{
    //! \name Multiple scattering
    bool displaced;
    real_type range_factor;
    MscStepLimitAlgorithm step_limit_algorithm;
    //!@}

    //! Default options for light particles (electrons/positrons)
    static ParticleOptions default_light()
    {
        ParticleOptions opts;
        using namespace celeritas::units::literals;
        opts.min_range = 1.0_mm;
        opts.max_step_over_range = 0.2;
        opts.lowest_energy = ParticleOptions::Energy{0.001};
        opts.displaced = true;
        opts.range_factor = 0.04;
        opts.step_limit_algorithm = MscStepLimitAlgorithm::safety;
        return opts;
    };

    //! Default options for heavy particles (muons/hadrons)
    static ParticleOptions default_heavy()
    {
        ParticleOptions opts;
        using namespace celeritas::units::literals;
        opts.min_range = 0.1_mm;
        opts.max_step_over_range = 0.2;
        opts.lowest_energy = ParticleOptions::Energy{0.001};
        opts.displaced = false;
        opts.range_factor = 0.2;
        opts.step_limit_algorithm = MscStepLimitAlgorithm::minimal;
        return opts;
    };
};

//---------------------------------------------------------------------------//
/*!
 * Physics configuration options.
 *
 * Input options are:
 * - \c fixed_step_limiter: if nonzero, prevent any tracks from taking a step
 *   longer than this length.
 * - \c min_eprime_over_e: energy scaling fraction used to estimate the maximum
 *   cross section over the step in the integral approach for energy loss
 *   processes.
 * - \c linear_loss_limit: if the mean energy loss along a step is greater than
 *   this fractional value of the pre-step kinetic energy, recalculate the
 *   energy loss.
 * - \c lambda_limit: limit on the MSC mean free path.
 * - \c safety_factor: used in the MSC step limitation algorithm to restrict
 *   the step size to \f$ f_s s \f$, where \f$ f_s \f$ is the safety factor and
 *   \f$  s \f$ is the safety distance.
 * - \c secondary_stack_factor: the number of secondary slots per track slot
 *   allocated.
 * - \c disable_integral_xs: for particles with energy loss processes, the
 *   particle energy changes over the step, so the assumption that the cross
 *   section is constant is no longer valid. By default, many charged particle
 *   processes use MC integration to sample the discrete interaction length
 *   with the correct probability. Disable this integral approach for all
 *   processes.
 */
struct PhysicsOptions
{
    //!@{
    //! \name Range calculation
    real_type fixed_step_limiter{0};
    //!@}

    //!@{
    //! \name Energy loss
    real_type min_eprime_over_e{0.8};
    real_type linear_loss_limit{0.01};
    //!@}

    //!@{
    //! \name Multiple scattering
    real_type lambda_limit{real_type{1} * units::millimeter};
    real_type safety_factor{0.6};
    //!@}

    //!@{
    //! \name Particle-dependent parameters
    ParticleOptions light{ParticleOptions::default_light()};
    ParticleOptions heavy{ParticleOptions::default_heavy()};
    //!@}

    real_type secondary_stack_factor{3};
    bool disable_integral_xs{false};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
