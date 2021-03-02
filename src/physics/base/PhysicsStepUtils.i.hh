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
#include "physics/grid/EnergyLossCalculator.hh"
#include "physics/grid/InverseRangeCalculator.hh"
#include "physics/grid/RangeCalculator.hh"
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

    constexpr real_type inf = numeric_limits<real_type>::infinity();
    using VGT               = ValueGridType;

    // Loop over all processes that apply to this track (based on particle
    // type) and calculate cross section and particle range.
    real_type total_macro_xs = 0;
    real_type min_range      = inf;
    for (auto ppid : range(ParticleProcessId{physics.num_particle_processes()}))
    {
        real_type               process_xs = 0;
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
            auto calc_xs = physics.make_calculator<XsCalculator>(grid_id);
            process_xs = calc_xs(particle.energy());
            total_macro_xs += process_xs;
        }
        physics.per_process_xs(ppid) = process_xs;

        if (auto grid_id = physics.value_grid(VGT::range, ppid))
        {
            auto calc_range = physics.make_calculator<RangeCalculator>(grid_id);
            real_type process_range = calc_range(particle.energy());
            min_range = min(min_range, process_range);
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
 * Geant3 manual (1993).
 *
 * In Geant4's calculation \c G4VEnergyLossProcess::AlongStepDoIt:
 * - \c reduceFactor is 1 except for heavy ions
 * - \c massRatio is 1 except for heavy ions
 * - \c length = step
 * - \c fRange = physics.range_limit()
 * - \c linLossLimit = \c linear_loss_limit = \f$xi\f$
 *
 * Energy loss rate is a function of differential cross section: the integral
 * of low-energy secondaries (below \c T) produced as a function of energy: \f[
 *   \frac{dE}{dx} = N_Z \int_0^T \frac{d \sigma_Z(E, T)}{dT} T dT
 * \f]
 *
 * The stopping range \em R due to these low-energy processes is:
 * \f[
 *   R = \int_0 ^{E_0} - \frac{dx}{dE} dE .
 * \f]
 *
 * Both Celeritas and Geant4 approximate the range limit as the minimum range
 * over all processes, rather than the range as a result from integrating all
 * energy loss processes over the allowed energy range.
 *
 * Geant4's stepping algorithm independently stores \c fRange for each process,
 * then (looping externally over all processes) calculates energy loss, checks
 * for the linear loss limit, and reduces the particle energy. Celeritas
 * inverts this loop so the total energy loss from along-step processess (not
 * including multiple scattering) is calculated first, then checked against
 * being greater than the lineaer loss limit.
 *
 * \todo The geant3 manual makes the point that linear interpolation of energy
 * loss rate results in a piecewise constant energy deposition curve, which is
 * why they use spline interpolation. Investigate higher-order reconstruction
 * of energy loss curve, e.g. through spline-based interpolation or log-log
 * interpolation.
 *
 * \note The inverse range correction assumes range is always the integral of
 * the stopping power/energy loss.
 */
CELER_FUNCTION ParticleTrackView::Energy
               calc_energy_loss(const ParticleTrackView& particle,
                                const PhysicsTrackView&  physics,
                                real_type                step)
{
    CELER_EXPECT(step >= 0);
    static_assert(ParticleTrackView::Energy::unit_type::value()
                      == EnergyLossCalculator::Energy::unit_type::value(),
                  "Incompatible energy types");

    using VGT = ValueGridType;
    const auto pre_step_energy = particle.energy();

    // Calculate the sum of energy loss rate over all processes.
    real_type total_eloss_rate = 0;
    for (auto ppid : range(ParticleProcessId{physics.num_particle_processes()}))
    {
        if (auto grid_id = physics.value_grid(VGT::energy_loss, ppid))
        {
            auto calc_eloss_rate
                = physics.make_calculator<EnergyLossCalculator>(grid_id);
            total_eloss_rate += calc_eloss_rate(pre_step_energy);
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
        for (auto ppid :
             range(ParticleProcessId{physics.num_particle_processes()}))
        {
            if (auto grid_id = physics.value_grid(VGT::range, ppid))
            {
                // Recalculate beginning-of-step range (instead of storing)
                auto calc_range
                    = physics.make_calculator<RangeCalculator>(grid_id);
                real_type remaining_range = calc_range(pre_step_energy) - step;
                CELER_ASSERT(remaining_range > 0);

                // Calculate energy along the range curve corresponding to the
                // actual step taken: this gives the exact energy loss over the
                // step due to this process.
                auto calc_energy
                    = physics.make_calculator<InverseRangeCalculator>(grid_id);
                eloss += (pre_step_energy.value()
                          - calc_energy(remaining_range).value());
            }
        }
        CELER_ASSERT(eloss > 0);
        CELER_ASSERT(eloss < pre_step_energy.value());
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
 */
template<class Engine>
CELER_FUNCTION ModelId select_model(const ParticleTrackView& particle,
                                    const PhysicsTrackView&  physics,
                                    Engine&                  rng)
{
    if (physics.interaction_mfp() > 0)
    {
        // Nonzero MFP to interaction -- no interaction model
        return {};
    }

    // Sample ParticleProcessId from physics.per_process_xs()
    (void)sizeof(rng);

    // Get ModelGroup from physics.models(ppid);

    // Find ModelId corresponding to energy bin
    (void)sizeof(particle.energy());

    CELER_NOT_IMPLEMENTED("selecting a model for interaction");
}

//---------------------------------------------------------------------------//
} // namespace celeritas
