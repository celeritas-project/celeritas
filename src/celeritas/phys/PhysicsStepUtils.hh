//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsStepUtils.hh
//! \brief Helper functions for physics step processing
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "corecel/random/distribution/Selector.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/EnergyLossCalculator.hh"
#include "celeritas/grid/InverseRangeCalculator.hh"
#include "celeritas/grid/RangeCalculator.hh"
#include "celeritas/grid/SplineCalculator.hh"
#include "celeritas/grid/XsCalculator.hh"
#include "celeritas/mat/MaterialTrackView.hh"

#include "ParticleTrackView.hh"
#include "PhysicsStepView.hh"
#include "PhysicsTrackView.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Sample the process for the discrete interaction.
 */
template<class Engine>
inline CELER_FUNCTION ParticleProcessId
find_ppid(MaterialView const& material,
          ParticleTrackView const& particle,
          PhysicsTrackView const& physics,
          PhysicsStepView& pstep,
          Engine& rng)
{
    if (physics.at_rest_process() && particle.is_stopped())
    {
        // If the particle is stopped and has an at-rest process, select it
        return physics.at_rest_process();
    }

    // Sample the process from the pre-calculated per-process cross section
    ParticleProcessId ppid = celeritas::make_selector(
        [&pstep](ParticleProcessId ppid) { return pstep.per_process_xs(ppid); },
        ParticleProcessId{physics.num_particle_processes()},
        pstep.macro_xs())(rng);
    CELER_ASSERT(ppid);

    // Determine if the discrete interaction occurs for particles with energy
    // loss processes
    if (physics.integral_xs_process(ppid))
    {
        // Recalculate the cross section at the post-step energy \f$ E_1 \f$
        real_type xs = physics.calc_xs(ppid, material, particle.energy());

        // The discrete interaction occurs with probability \f$ \sigma(E_1) /
        // \sigma_{\max} \f$. Note that it's possible for \f$ \sigma(E_1) \f$
        // to be larger than the estimate of the maximum cross section over the
        // step \f$ \sigma_{\max} \f$.
        if (generate_canonical(rng) * pstep.per_process_xs(ppid) > xs)
        {
            // No interaction occurs; reset the physics state and continue
            // tracking
            return {};
        }
    }
    return ppid;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Calculate physics step limits based on cross sections and range limiters.
 */
inline CELER_FUNCTION StepLimit
calc_physics_step_limit(MaterialTrackView const& material,
                        ParticleTrackView const& particle,
                        PhysicsTrackView& physics,
                        PhysicsStepView& pstep)
{
    CELER_EXPECT(physics.has_interaction_mfp());

    /*! \todo For particles with decay, macro XS calculation will incorporate
     * decay probability, dividing decay constant by speed to become 1/len to
     * compete with interactions.
     *
     * \todo For neutral particles that haven't changed material since the last
     * step, we can reuse the previously calculated cross section.
     */

    // Loop over all processes that apply to this track (based on particle
    // type) and calculate cross section and particle range.
    real_type total_macro_xs = 0;
    for (auto ppid : range(ParticleProcessId{physics.num_particle_processes()}))
    {
        real_type process_xs = 0;
        if (auto const& process = physics.integral_xs_process(ppid))
        {
            // If the integral approach is used and this particle has an energy
            // loss process, estimate the maximum cross section over the step
            process_xs = physics.calc_max_xs(
                process, ppid, material.material_record(), particle.energy());
        }
        else
        {
            // Calculate the macroscopic cross section for this process
            process_xs = physics.calc_xs(
                ppid, material.material_record(), particle.energy());
        }
        // Accumulate process cross section into the total cross section and
        // save it for later
        total_macro_xs += process_xs;
        pstep.per_process_xs(ppid) = process_xs;
    }
    pstep.macro_xs(total_macro_xs);
    CELER_ASSERT(total_macro_xs > 0 || !particle.is_stopped());

    // Determine limits from discrete interactions
    StepLimit limit;
    limit.action = physics.scalars().discrete_action();
    if (particle.is_stopped())
    {
        limit.step = 0;
    }
    else
    {
        limit.step = physics.interaction_mfp() / total_macro_xs;
        if (auto grid_id = physics.range_grid())
        {
            auto calc_range = physics.make_calculator<RangeCalculator>(grid_id);
            real_type range = calc_range(particle.energy());
            // Save range for the current step and reuse it elsewhere
            physics.dedx_range(range);

            // Convert to the scaled range
            real_type eloss_step = physics.range_to_step(range);
            if (eloss_step <= limit.step)
            {
                limit.step = eloss_step;
                limit.action = physics.scalars().range_action();
            }

            // Limit charged particle step size
            real_type fixed_limit = physics.scalars().fixed_step_limiter;
            if (fixed_limit > 0 && fixed_limit < limit.step)
            {
                limit.step = fixed_limit;
                limit.action = physics.scalars().fixed_step_action;
            }
        }
        else if (physics.num_particle_processes() == 0)
        {
            // Clear post-step action so that unknown particles don't interact
            limit.action = {};
        }
    }

    return limit;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate mean energy loss over the given "true" step length.
 *
 * Stopping power is an integral over low-exiting-energy
 * secondaries. Above some threshold energy \em T_c we treat exiting
 * secondaries discretely; below it, we lump them into this continuous loss
 * term that varies based on the energy, the atomic number density, and the
 * element number:
 * \f[
 *   \frac{dE}{dx} = N_Z \int_0^{T_c} \frac{d \sigma_Z(E, T)}{dT} T dT
 * \f]
 * Here, the cross section is a function of the primary's energy \em E and the
 * exiting secondary energy \em T.
 *
 * The stopping range \em R due to these low-energy processes is an integral
 * over the inverse of the stopping power: basically the distance that will
 * take a particle (without discrete processes at play) from its current energy
 * to an energy of zero.
 * \f[
 *   R = \int_0 ^{E_0} - \frac{dx}{dE} dE .
 * \f]
 *
 * Both Celeritas and Geant4 approximate the range limit as the minimum range
 * over all processes, rather than the range as a result from integrating all
 * energy loss processes over the allowed energy range. This is usually not
 * a problem in practice because the range will get automatically decreased by
 * \c range_to_step , and the above range calculation neglects energy loss by
 * discrete processes.
 *
 * Both Geant4 and Celeritas integrate the energy loss terms across processes
 * to get a single energy loss vector per particle type. The range table is an
 * integral of the mean stopping power: the total distance for the particle's
 * energy to reach zero.
 *
 * \note The inverse range correction assumes range is always the integral of
 * the stopping power/energy loss.
 *
 * \todo The GEANT3 manual \cite{geant3-1993} makes the point that linear
 * interpolation of energy
 * loss rate results in a piecewise constant energy deposition curve, which is
 * why they use spline interpolation. Investigate higher-order reconstruction
 * of energy loss curve, e.g. through spline-based interpolation or log-log
 * interpolation.
 *
 * \note See section 7.2.4 Run Time Energy Loss Computation of the Geant4
 * physics manual \cite{g4prm}. See also the longer discussions in section 8
 * of PHYS010 of the Geant3 manual.
 *
 * Zero energy loss can occur in the following cases:
 * - The energy loss value at the given energy is zero (e.g. high energy
 * particles)
 * - The urban model is selected and samples zero collisions (possible in thin
 * materials and/or small steps)
 */
inline CELER_FUNCTION ParticleTrackView::Energy
calc_mean_energy_loss(ParticleTrackView const& particle,
                      PhysicsTrackView const& physics,
                      real_type step)
{
    CELER_EXPECT(step > 0);
    using Energy = ParticleTrackView::Energy;

    Energy const pre_step_energy = particle.energy();

    // Calculate the sum of energy loss rate over all processes.
    Energy eloss;
    {
        auto grid_id = physics.energy_loss_grid();
        CELER_ASSERT(grid_id);

        auto calc_eloss_rate
            = physics.make_calculator<EnergyLossCalculator>(grid_id);
        eloss = Energy{step * calc_eloss_rate(pre_step_energy)};
    }

    if (eloss >= pre_step_energy * physics.scalars().linear_loss_limit)
    {
        // Enough energy is lost over this step that the dE/dx linear
        // approximation is probably wrong. Use the definition of the range as
        // the integral of 1/loss to back-calculate the actual energy loss
        // along the curve given the actual step.
        auto grid_id = physics.range_grid();
        CELER_ASSERT(grid_id);

        // Use the range limit stored from calc_physics_step_limit
        real_type range = physics.dedx_range();
        if (step == range)
        {
            // NOTE: eloss should be pre_step_energy at this point only if the
            // range was the step limiter (step == range), and if the
            // range-to-step conversion was 1.
            return pre_step_energy;
        }
        CELER_ASSERT(range > step);

        // Calculate energy along the range curve corresponding to the actual
        // step taken: this gives the exact energy loss over the step due to
        // this process. If the step is very near the range (a few ULP off, for
        // example), then the post-step energy will be calculated as zero
        // without going through the condition above.
        auto calc_energy = physics.make_calculator<InverseRangeCalculator>(
            physics.inverse_range_grid());
        eloss = pre_step_energy - calc_energy(range - step);

        // Spline interpolation does not ensure roundtrip consistency between
        // the range and the inverse. This can lead to negative values for the
        // energy loss
        eloss = clamp_to_nonneg(eloss);
    }

    CELER_ENSURE(eloss >= zero_quantity());
    return eloss;
}

//---------------------------------------------------------------------------//
/*!
 * Choose the physics model for a track's pending interaction.
 *
 * - If the interaction MFP is zero, the particle is undergoing a discrete
 *   interaction. Otherwise, the result is a false ActionId.
 * - Sample from the previously calculated per-process cross section/decay to
 *   determine the interacting process ID.
 * - From the process ID and (post-slowing-down) particle energy, we obtain the
 *   applicable model ID.
 * - For energy loss (continuous-discrete) processes, the post-step energy will
 *   be different from the pre-step energy, so the assumption that the cross
 *   section is constant along the step is no longer valid. Use the "integral
 *   approach" to sample the discrete interaction from the correct probability
 *   distribution (section 7.4 of the Geant4 Physics Reference release 10.6).
 */
template<class Engine>
CELER_FUNCTION ActionId
select_discrete_interaction(MaterialView const& material,
                            ParticleTrackView const& particle,
                            PhysicsTrackView const& physics,
                            PhysicsStepView& pstep,
                            Engine& rng)
{
    // Nonzero MFP to interaction -- no interaction model
    CELER_EXPECT(physics.interaction_mfp() <= 0);
    CELER_EXPECT(pstep.macro_xs() > 0);

    // Sample the process from the pre-calculated cross sections
    auto ppid = find_ppid(material, particle, physics, pstep, rng);
    if (!ppid)
    {
        return physics.scalars().integral_rejection_action();
    }

    // Find the model that applies at the particle energy
    auto find_model = physics.make_model_finder(ppid);
    auto pmid = find_model(particle.energy());

    ElementComponentId elcomp_id{};
    if (material.num_elements() == 1)
    {
        elcomp_id = ElementComponentId{0};
    }
    else if (auto table_id = physics.cdf_table(pmid))
    {
        // Sample an element for discrete interactions that require it and for
        // materials with more than one element
        auto select_element
            = physics.make_element_selector(table_id, particle.energy());
        elcomp_id = select_element(rng);
    }
    pstep.element(elcomp_id);

    return physics.model_to_action(physics.model_id(pmid));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
