//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsStepUtils.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"
#include "base/Algorithms.hh"
#include "base/NumericLimits.hh"
#include "base/Range.hh"
#include "random/distributions/BernoulliDistribution.hh"
#include "random/distributions/GenerateCanonical.hh"
#include "random/Selector.hh"
#include "physics/em/EnergyLossDistribution.hh"
#include "physics/grid/EnergyLossCalculator.hh"
#include "physics/grid/InverseRangeCalculator.hh"
#include "physics/grid/RangeCalculator.hh"
#include "physics/grid/ValueGridData.hh"
#include "physics/grid/XsCalculator.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate physics step limits based on cross sections and range limiters.
 */
inline CELER_FUNCTION real_type
calc_tabulated_physics_step(const MaterialTrackView& material,
                            const ParticleTrackView& particle,
                            PhysicsTrackView&        physics)
{
    CELER_EXPECT(physics.has_interaction_mfp());

    const real_type inf = numeric_limits<real_type>::infinity();
    using VGT           = ValueGridType;

    // Loop over all processes that apply to this track (based on particle
    // type) and calculate cross section and particle range.
    real_type total_macro_xs = 0;
    real_type min_range      = inf;
    for (auto ppid : range(ParticleProcessId{physics.num_particle_processes()}))
    {
        real_type process_xs = 0;
        if (auto model_id = physics.hardwired_model(ppid, particle.energy()))
        {
            // Calculate macroscopic cross section on the fly for special
            // hardwired processes.
            auto material_view = material.material_view();
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

        if (auto grid_id = physics.value_grid(VGT::range, ppid))
        {
            auto calc_range = physics.make_calculator<RangeCalculator>(grid_id);
            real_type process_range = calc_range(particle.energy());
            min_range               = min(min_range, process_range);
        }
    }
    physics.macro_xs(total_macro_xs);

    if (min_range != inf)
    {
        // One or more range limiters applied: scale range limit according to
        // user options
        min_range = physics.range_to_step(min_range);
    }

    // Update step length with discrete interaction
    return min(min_range, physics.interaction_mfp() / total_macro_xs);
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
 * Geant4's stepping algorithm independently stores the range for each process,
 * then (looping externally over all processes) calculates energy loss, checks
 * for the linear loss limit, and reduces the particle energy. Celeritas
 * inverts this loop so the total energy loss from along-step processess (not
 * including multiple scattering) is calculated first, then checked against
 * being greater than the linear loss limit.
 *
 * If energy loss is greater than the loss limit, we loop over all
 * processes with range tables and recalculate the pre-step range and solve for
 * the exact post-step energy loss.
 *
 * \note The inverse range correction assumes range is always the integral of
 * the stopping power/energy loss.
 *
 * \todo The geant3 manual makes the point that linear interpolation of energy
 * loss rate results in a piecewise constant energy deposition curve, which is
 * why they use spline interpolation. Investigate higher-order reconstruction
 * of energy loss curve, e.g. through spline-based interpolation or log-log
 * interpolation.
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
    CELER_EXPECT(step >= 0);
    static_assert(ParticleTrackView::Energy::unit_type::value()
                      == EnergyLossCalculator::Energy::unit_type::value(),
                  "Incompatible energy types");

    using VGT                  = ValueGridType;
    const auto pre_step_energy = particle.energy();

    // Calculate the sum of energy loss rate over all processes.
    real_type total_eloss_rate = 0;
    if (auto ppid = physics.eloss_ppid())
    {
        if (auto grid_id = physics.value_grid(VGT::energy_loss, ppid))
        {
            auto calc_eloss_rate
                = physics.make_calculator<EnergyLossCalculator>(grid_id);
            total_eloss_rate = calc_eloss_rate(pre_step_energy);
        }
    }

    // Scale loss rate by step length
    real_type eloss = total_eloss_rate * step;

    if (eloss > pre_step_energy.value() * physics.linear_loss_limit())
    {
        // Enough energy is lost over this step that the dE/dx linear
        // approximation is probably wrong. Use the definition of the range as
        // the integral of 1/loss to back-calculate the actual energy loss
        // along the curve given the actual step.
        eloss = 0;
        if (auto ppid = physics.eloss_ppid())
        {
            if (auto grid_id = physics.value_grid(VGT::range, ppid))
            {
                // Recalculate beginning-of-step range (instead of storing)
                auto calc_range
                    = physics.make_calculator<RangeCalculator>(grid_id);
                real_type remaining_range = calc_range(pre_step_energy) - step;
                CELER_ASSERT(remaining_range >= 0);

                // Calculate energy along the range curve corresponding to the
                // actual step taken: this gives the exact energy loss over the
                // step due to this process.
                auto calc_energy
                    = physics.make_calculator<InverseRangeCalculator>(grid_id);
                eloss = (pre_step_energy.value()
                         - calc_energy(remaining_range).value());
            }
        }
        CELER_ASSERT(eloss > 0);
        CELER_ASSERT(eloss <= pre_step_energy.value());
    }

    // Add energy loss fluctuations if this is the "energy loss" process
    if (eloss > 0 && eloss < pre_step_energy.value()
        && physics.add_fluctuation())
    {
        EnergyLossDistribution sample_loss(physics.fluctuation(),
                                           cutoffs,
                                           material,
                                           particle,
                                           units::MevEnergy{eloss},
                                           step);
        eloss = min(sample_loss(rng).value(), pre_step_energy.value());
    }

    CELER_ENSURE(eloss <= pre_step_energy.value());
    return ParticleTrackView::Energy{eloss};
}

//---------------------------------------------------------------------------//
/*!
 * Choose the physics model for a track's pending interaction.
 *
 * - If the interaction MFP is zero, the particle is undergoing a discrete
 *   interaction. Otherwise, the result is a false ModelId.
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
CELER_FUNCTION ProcessIdModelId select_process_and_model(
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
    return ProcessIdModelId{ppid, model_id};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
