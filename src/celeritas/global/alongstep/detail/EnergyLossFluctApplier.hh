//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/EnergyLossFluctApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/distribution/EnergyLossHelper.hh"
#include "celeritas/em/distribution/EnergyLossTraits.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/PhysicsStepUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply energy loss (with fluctuations) to a track.
 */
class EnergyLossFluctApplier
{
  public:
    //!@{
    //! \name Type aliases
    using Energy    = EnergyLossHelper::Energy;
    using ParamsRef = NativeCRef<FluctuationData>;
    //!@}

  public:
    //! Construct from reference to data
    explicit CELER_FUNCTION EnergyLossFluctApplier(const ParamsRef& params)
        : fluct_params_{params}
    {
    }

    // Apply to the track
    inline CELER_FUNCTION void
    operator()(CoreTrackView const& track, StepLimit* step_limit);

  private:
    //// DATA ////

    //! Reference to fluctuation data
    const ParamsRef& fluct_params_;

    //// HELPER FUNCTIONS ////

    template<EnergyLossFluctuationModel M>
    CELER_FUNCTION Energy sample_energy_loss(const EnergyLossHelper& helper,
                                             Energy                  max_loss,
                                             RngEngine&              rng);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply energy loss to the given track.
 */
CELER_FUNCTION void
EnergyLossFluctApplier::operator()(CoreTrackView const& track,
                                   StepLimit*           step_limit)
{
    CELER_EXPECT(step_limit->step > 0);

    auto phys = track.make_physics_view();
    if (!phys.eloss_ppid())
    {
        // No energy loss process for this particle type
        return;
    }

    Energy eloss;

    auto particle = track.make_particle_view();
    if (particle.energy() < phys.scalars().eloss_calc_limit
        && step_limit->action != track.boundary_action())
    {
        // Immediately stop low-energy tracks (as long as they're not crossing
        // a boundary)
        // TODO: this should happen before creating tracks from secondaries
        // *OR* after slowing down tracks: duplicated in
        // EnergyLossApplier.hh
        eloss = particle.energy();
    }
    else
    {
        // Calculate mean energy loss
        eloss = calc_mean_energy_loss(particle, phys, step_limit->step);

        if (fluct_params_ && eloss > zero_quantity()
            && eloss < particle.energy())
        {
            // Apply energy loss fluctuations
            auto cutoffs  = track.make_cutoff_view();
            auto material = track.make_material_view();

            EnergyLossHelper loss_helper(fluct_params_,
                                         cutoffs,
                                         material,
                                         particle,
                                         eloss,
                                         step_limit->step);

            auto rng = track.make_rng_engine();
            switch (loss_helper.model())
            {
#define ASU_SAMPLE_ELOSS(MODEL)                                              \
    case EnergyLossFluctuationModel::MODEL:                                  \
        eloss = this->sample_energy_loss<EnergyLossFluctuationModel::MODEL>( \
            loss_helper, particle.energy(), rng);                            \
        break
                ASU_SAMPLE_ELOSS(none);
                ASU_SAMPLE_ELOSS(gamma);
                ASU_SAMPLE_ELOSS(gaussian);
                ASU_SAMPLE_ELOSS(urban);
#undef ASU_SAMPLE_ELOSS
            }
        }
    }

    CELER_ASSERT(eloss <= particle.energy());
    if (eloss == particle.energy())
    {
        // Particle lost all energy over the step
        if (CELER_UNLIKELY(step_limit->action == track.boundary_action()))
        {
            // Particle lost all energy *and* is at a geometry boundary.
            // It therefore physically moved too far over the step, since
            // the range is supposed to be the integral of the inverse
            // energy loss rate. Bump particle slightly away from boundary
            // to avoid on-surface initialization/direction change.
            real_type backward_bump = real_type(-1e-5) * step_limit->step;
            // Force the step limiter to be "range"
            step_limit->action = phys.scalars().range_action();
            step_limit->step += backward_bump;

            auto  geo = track.make_geo_view();
            Real3 pos = geo.pos();
            axpy(backward_bump, geo.dir(), &pos);
            geo.move_internal(pos);
        }

        if (!phys.has_at_rest())
        {
            // Immediately kill stopped particles with no at rest processes
            auto sim = track.make_sim_view();
            sim.status(TrackStatus::killed);
        }
        else
        {
            // Particle slowed down to zero: force a discrete interaction
            step_limit->action = phys.scalars().discrete_action();
        }
    }

    if (eloss > zero_quantity())
    {
        // Deposit energy loss
        auto step = track.make_physics_step_view();
        step.deposit_energy(eloss);
        particle.subtract_energy(eloss);
    }
}

//---------------------------------------------------------------------------//
template<EnergyLossFluctuationModel M>
CELER_FUNCTION auto
EnergyLossFluctApplier::sample_energy_loss(const EnergyLossHelper& helper,
                                           Energy                  max_loss,
                                           RngEngine& rng) -> Energy
{
    CELER_EXPECT(helper.model() == M);

    using Distribution = typename EnergyLossTraits<M>::type;

    Distribution sample_eloss{helper};
    Energy       result = sample_eloss(rng);

    // TODO: investigate cases where sampled energy loss is greater than
    // the track's actual energy, i.e. the range limiter failed.
    result = Energy{celeritas::min(result.value(), max_loss.value())};
    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
