//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsStepUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Algorithms.hh"
#include "base/Assert.hh"
#include "base/NumericLimits.hh"
#include "base/Range.hh"
#include "base/Types.hh"
#include "physics/em/EnergyLossDistribution.hh"
#include "physics/grid/EnergyLossCalculator.hh"
#include "physics/grid/InverseRangeCalculator.hh"
#include "physics/grid/RangeCalculator.hh"
#include "physics/grid/ValueGridData.hh"
#include "physics/grid/XsCalculator.hh"
#include "physics/material/MaterialTrackView.hh"
#include "random/Selector.hh"
#include "random/distributions/BernoulliDistribution.hh"
#include "random/distributions/GenerateCanonical.hh"

#include "CutoffView.hh"
#include "ParticleTrackView.hh"
#include "PhysicsTrackView.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
namespace
{
//---------------------------------------------------------------------------//
template<EnergyLossFluctuationModel M, class Engine>
CELER_FUNCTION EnergyLossHelper::Energy
               sample_energy_loss(const EnergyLossHelper&  helper,
                                  EnergyLossHelper::Energy max_loss,
                                  Engine&                  rng)
{
    using Energy        = EnergyLossHelper::Energy;
    auto   sample_eloss = make_distribution<M>(helper);
    Energy result       = sample_eloss(rng);

    // TODO: investigate cases where sampled energy loss is greater than
    // the track's actual energy, i.e. the range limiter failed.
    result = Energy{celeritas::min(result.value(), max_loss.value())};
    return result;
}
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Calculate physics step limits based on cross sections and range limiters.
 */
inline CELER_FUNCTION StepLimit
calc_physics_step_limit(const MaterialTrackView& material,
                        const ParticleTrackView& particle,
                        PhysicsTrackView&        physics)
{
    CELER_EXPECT(physics.has_interaction_mfp());

    using VGT = ValueGridType;

    // TODO: for particles with decay, macro XS calculation will incorporate
    // decay probability, dividing decay constant by speed to become 1/cm to
    // compete with interactions

    // Loop over all processes that apply to this track (based on particle
    // type) and calculate cross section and particle range.
    real_type total_macro_xs = 0;
    for (auto ppid : range(ParticleProcessId{physics.num_particle_processes()}))
    {
        real_type process_xs = 0;
        if (auto model_id = physics.hardwired_model(ppid, particle.energy()))
        {
            // Calculate macroscopic cross section on the fly for special
            // hardwired processes.
            auto material_view = material.make_material_view();
            process_xs         = physics.calc_xs_otf(
                model_id, material_view, particle.energy());
            total_macro_xs += process_xs;
        }
        else if (auto grid_id = physics.value_grid(VGT::macro_xs, ppid))
        {
            // Calculate macroscopic cross section for this process, then
            // accumulate it into the total cross section and save the cross
            // section for later.
            process_xs = physics.calc_xs(ppid, grid_id, particle.energy());
            total_macro_xs += process_xs;
        }
        physics.per_process_xs(ppid) = process_xs;
    }
    physics.macro_xs(total_macro_xs);

    // Determine limits from discrete interactions
    StepLimit limit;
    if (particle.is_stopped())
    {
        limit.step   = 0;
        limit.action = physics.scalars().discrete_action();
    }
    else
    {
        limit.step   = physics.interaction_mfp() / total_macro_xs;
        limit.action = physics.scalars().discrete_action();
    }

    if (auto ppid = physics.eloss_ppid())
    {
        auto grid_id    = physics.value_grid(VGT::range, ppid);
        auto calc_range = physics.make_calculator<RangeCalculator>(grid_id);
        real_type range = calc_range(particle.energy());
        // TODO: save range ?
        real_type eloss_step = physics.range_to_step(range);
        if (eloss_step <= limit.step)
        {
            limit.step   = eloss_step;
            limit.action = physics.scalars().range_action();
        }
    }

    return limit;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate mean energy loss over the given "true" step length.
 *
 * See section 7.2.4 Run Time Energy Loss Computation of the Geant4 physics
 * manual. See also the longer discussions in section 8 of PHYS010 of the
 * Geant3 manual (1993). Stopping power is an integral over low-exiting-energy
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
 * \todo The geant3 manual makes the point that linear interpolation of energy
 * loss rate results in a piecewise constant energy deposition curve, which is
 * why they use spline interpolation. Investigate higher-order reconstruction
 * of energy loss curve, e.g. through spline-based interpolation or log-log
 * interpolation.
 *
 * Zero energy loss can occur in the following cases:
 * - The particle doesn't have slowing-down energy loss (e.g. photons)
 * - The energy loss value at the given energy is zero (e.g. high energy
 * particles)
 * - The urban model is selected and samples zero collisions (possible in thin
 * materials and/or small steps)
 */
template<class Engine>
CELER_FUNCTION ParticleTrackView::Energy
               calc_energy_loss(const CutoffView&        cutoffs,
                                const MaterialTrackView& material,
                                const ParticleTrackView& particle,
                                const PhysicsTrackView&  physics,
                                real_type                step,
                                Engine&                  rng)
{
    CELER_EXPECT(step > 0);
    using Energy = ParticleTrackView::Energy;
    using VGT    = ValueGridType;
    static_assert(Energy::unit_type::value()
                      == EnergyLossCalculator::Energy::unit_type::value(),
                  "Incompatible energy types");

    auto ppid = physics.eloss_ppid();
    if (!ppid)
    {
        // No energy loss processes for this particle
        return Energy{0};
    }

    const real_type pre_step_energy = value_as<Energy>(particle.energy());

    // Calculate the sum of energy loss rate over all processes.
    real_type eloss;
    {
        auto grid_id = physics.value_grid(VGT::energy_loss, ppid);
        CELER_ASSERT(grid_id);
        auto calc_eloss_rate
            = physics.make_calculator<EnergyLossCalculator>(grid_id);
        eloss = step * calc_eloss_rate(Energy{pre_step_energy});
    }

    if (eloss >= pre_step_energy * physics.linear_loss_limit())
    {
        // Enough energy is lost over this step that the dE/dx linear
        // approximation is probably wrong. Use the definition of the range as
        // the integral of 1/loss to back-calculate the actual energy loss
        // along the curve given the actual step.
        auto grid_id = physics.value_grid(VGT::range, ppid);
        CELER_ASSERT(grid_id);

        // Recalculate beginning-of-step range (instead of storing)
        auto calc_range = physics.make_calculator<RangeCalculator>(grid_id);
        real_type range = calc_range(Energy{pre_step_energy});
        if (step == range)
        {
            // TODO: eloss should be pre_step_energy at this point only if the
            // range was the step limiter (step == range), and if the
            // range-to-step conversion was 1.
            return Energy{pre_step_energy};
        }
        CELER_ASSERT(range > step);

        // Calculate energy along the range curve corresponding to the actual
        // step taken: this gives the exact energy loss over the step due to
        // this process. If the step is very near the range (a few ULP off, for
        // example), then the post-step energy will be calculated as zero
        // without going through the condition above.
        auto calc_energy
            = physics.make_calculator<InverseRangeCalculator>(grid_id);
        eloss = pre_step_energy - value_as<Energy>(calc_energy(range - step));
    }

    if (physics.add_fluctuation() && eloss > 0 && eloss < pre_step_energy)
    {
        EnergyLossHelper loss_helper(physics.fluctuation(),
                                     cutoffs,
                                     material,
                                     particle,
                                     Energy{eloss},
                                     step);

        switch (loss_helper.model())
        {
#define PSU_SAMPLE_ELOSS(MODEL)                                    \
    case EnergyLossFluctuationModel::MODEL:                        \
        eloss = value_as<Energy>(                                  \
            sample_energy_loss<EnergyLossFluctuationModel::MODEL>( \
                loss_helper, Energy{pre_step_energy}, rng));       \
        break
            PSU_SAMPLE_ELOSS(none);
            PSU_SAMPLE_ELOSS(gamma);
            PSU_SAMPLE_ELOSS(gaussian);
            PSU_SAMPLE_ELOSS(urban);
#undef PSU_SAMPLE_ELOSS
        }
    }

    CELER_ASSERT(eloss >= 0 && eloss <= pre_step_energy);
    return Energy{eloss};
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
CELER_FUNCTION ActionId select_discrete_interaction(
    const ParticleTrackView& particle, PhysicsTrackView& physics, Engine& rng)
{
    // Nonzero MFP to interaction -- no interaction model
    CELER_EXPECT(physics.interaction_mfp() <= 0);

    // Sample ParticleProcessId from physics.per_process_xs()
    ParticleProcessId ppid = celeritas::make_selector(
        [&physics](ParticleProcessId ppid) {
            return physics.per_process_xs(ppid);
        },
        ParticleProcessId{physics.num_particle_processes()},
        physics.macro_xs())(rng);

    // Determine if the discrete interaction occurs for energy loss
    // processes
    if (physics.use_integral_xs(ppid))
    {
        // This is an energy loss process that was sampled for a
        // discrete interaction, so it will have macro xs tables
        auto grid_id = physics.value_grid(ValueGridType::macro_xs, ppid);
        CELER_ASSERT(grid_id);

        // Recalculate the cross section at the post-step energy \f$
        // E_1 \f$
        auto      calc_xs = physics.make_calculator<XsCalculator>(grid_id);
        real_type xs      = calc_xs(particle.energy());

        // The discrete interaction occurs with probability \f$ \sigma(E_1) /
        // \sigma_{\max} \f$. Note that it's possible for \f$ \sigma(E_1) \f$
        // to be larger than the estimate of the maximum cross section over the
        // step \f$ \sigma_{\max} \f$.
        if (generate_canonical(rng) > xs / physics.per_process_xs(ppid))
        {
            // No interaction occurs; reset the physics state and continue
            // tracking
            physics = {};
            return {};
        }
    }

    // Select the model and return; See doc above for details.
    auto find_model = physics.make_model_finder(ppid);
    auto model_id   = find_model(particle.energy());
    CELER_ENSURE(model_id);
    return physics.model_to_action(model_id);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
