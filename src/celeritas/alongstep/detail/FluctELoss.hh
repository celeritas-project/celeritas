//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/FluctELoss.hh
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
class FluctELoss
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = EnergyLossHelper::Energy;
    using ParamsRef = NativeCRef<FluctuationData>;
    //!@}

  public:
    // Construct with fluctuation data
    inline explicit CELER_FUNCTION FluctELoss(ParamsRef const& params);

    // Whether energy loss can be used for this track
    inline CELER_FUNCTION bool is_applicable(CoreTrackView const&) const;

    // Apply to the track
    inline CELER_FUNCTION Energy calc_eloss(CoreTrackView const& track,
                                            real_type step,
                                            bool apply_cut);

    //! Indicate that we can lose all energy before hitting the dE/dx range
    static CELER_CONSTEXPR_FUNCTION bool imprecise_range() { return true; }

  private:
    //// DATA ////

    //! Reference to fluctuation data
    ParamsRef const fluct_params_;

    //// HELPER FUNCTIONS ////

    template<EnergyLossFluctuationModel M>
    inline CELER_FUNCTION Energy
    sample_energy_loss(EnergyLossHelper const& helper, RngEngine& rng);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to fluctuation data.
 */
CELER_FUNCTION FluctELoss::FluctELoss(ParamsRef const& params)
    : fluct_params_{params}
{
    CELER_EXPECT(fluct_params_);
}

//---------------------------------------------------------------------------//
/*!
 * Whether energy loss is used for this track.
 */
CELER_FUNCTION bool FluctELoss::is_applicable(CoreTrackView const& track) const
{
    // The track can be marked as `errored` *within* the along-step kernel,
    // during propagation
    if (track.sim().status() == TrackStatus::errored)
        return false;

    // Energy loss grid ID is 'false'
    return static_cast<bool>(track.physics().energy_loss_grid());
}

//---------------------------------------------------------------------------//
/*!
 * Apply energy loss to the given track.
 *
 * - Before and after slowing down we apply a tracking cut to cull low-energy
 *   charged particles.
 * - If energy loss fluctuations are enabled, we apply those based on the mean
 *   energy loss.
 * - If the sampled energy loss is greater than or equal to the particle's
 *   energy, we reduce it to the particle energy (if energy cuts are to be
 *   applied) or to the mean energy loss (if cuts are prohibited due to this
 *   being a non-physics-based step).
 */
CELER_FUNCTION auto FluctELoss::calc_eloss(CoreTrackView const& track,
                                           real_type step,
                                           bool apply_cut) -> Energy
{
    CELER_EXPECT(step > 0);

    auto particle = track.particle();
    auto phys = track.physics();

    if (apply_cut && particle.energy() < phys.particle_scalars().lowest_energy)
    {
        // Deposit all energy immediately when we start below the tracking cut
        return particle.energy();
    }

    // Calculate mean energy loss
    auto eloss = calc_mean_energy_loss(particle, phys, step);

    if (eloss < particle.energy() && eloss > zero_quantity())
    {
        // Apply energy loss fluctuations
        auto cutoffs = track.cutoff();
        auto material = track.material();

        EnergyLossHelper loss_helper(
            fluct_params_, cutoffs, material, particle, eloss, step);

        auto rng = track.rng();
        switch (loss_helper.model())
        {
#define ASU_SAMPLE_ELOSS(MODEL)                                              \
    case EnergyLossFluctuationModel::MODEL:                                  \
        eloss = this->sample_energy_loss<EnergyLossFluctuationModel::MODEL>( \
            loss_helper, rng);                                               \
        break
            ASU_SAMPLE_ELOSS(none);
            ASU_SAMPLE_ELOSS(gamma);
            ASU_SAMPLE_ELOSS(gaussian);
            ASU_SAMPLE_ELOSS(urban);
#undef ASU_SAMPLE_ELOSS
        }

        if (eloss >= particle.energy())
        {
            // Sampled energy loss can be greater than actual remaining energy
            // because the range calculation is based on the *mean* energy
            // loss.
            if (apply_cut)
            {
                // Clamp to actual particle energy so that it stops
                eloss = particle.energy();
            }
            else
            {
                // Don't go to zero energy at geometry boundaries: just use the
                // mean loss which should be positive because this isn't a
                // range-limited step.
                eloss = loss_helper.mean_loss();
                CELER_ASSERT(eloss < particle.energy());
            }
        }
    }

    if (apply_cut
        && (particle.energy() - eloss <= phys.particle_scalars().lowest_energy))
    {
        // Deposit all energy when we end below the tracking cut
        return particle.energy();
    }

    CELER_ASSERT(eloss <= particle.energy());
    CELER_ENSURE(eloss != particle.energy() || apply_cut
                 || track.sim().post_step_action()
                        == phys.scalars().range_action());
    return eloss;
}

//---------------------------------------------------------------------------//
template<EnergyLossFluctuationModel M>
CELER_FUNCTION auto
FluctELoss::sample_energy_loss(EnergyLossHelper const& helper,
                               RngEngine& rng) -> Energy
{
    CELER_EXPECT(helper.model() == M);

    using Distribution = typename EnergyLossTraits<M>::type;

    Distribution sample_eloss{helper};
    return sample_eloss(rng);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
