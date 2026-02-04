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
 *
 * \warning Because particle range is the integral of the \em mean energy loss,
 * and this samples from a distribution, the sampled energy loss may be more
 * than the particle's energy! We take care not to end a particle's life on a
 * boundary, which is a nonphysical bias.
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

    // Apply to the track
    inline CELER_FUNCTION Energy operator()(CoreTrackView const& track);

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
 *
 * \todo The gamma and gaussian energy loss models are never called by
 * positrons/electrons, only by muons
 */
CELER_FUNCTION auto FluctELoss::operator()(CoreTrackView const& track) -> Energy
{
    auto particle = track.particle();
    auto phys = track.physics();
    auto sim = track.sim();

    // Calculate mean energy loss
    auto eloss = calc_mean_energy_loss(particle, phys, sim.step_length());
    if (eloss == zero_quantity())
    {
        return eloss;
    }

    // Apply energy loss fluctuations
    auto cutoffs = track.cutoff();
    auto material = track.material();
    EnergyLossHelper loss_helper(
        fluct_params_, cutoffs, material, particle, eloss, sim.step_length());

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
        // loss. To fix this, we would need to sample the range from a
        // distribution as well.
        if (track.geometry().is_on_boundary())
        {
            // Don't stop particles on geometry boundaries: just use the
            // mean loss which should be positive because this isn't a
            // range-limited step.
            eloss = loss_helper.mean_loss();
            CELER_ASSERT(eloss < particle.energy());
        }
        else
        {
            // Clamp to actual particle energy so that it stops
            eloss = particle.energy();
        }
    }

    return eloss;
}

//---------------------------------------------------------------------------//
template<EnergyLossFluctuationModel M>
CELER_FUNCTION auto
FluctELoss::sample_energy_loss(EnergyLossHelper const& helper, RngEngine& rng)
    -> Energy
{
    CELER_EXPECT(helper.model() == M);

    using Distribution = typename EnergyLossTraits<M>::type;

    Distribution sample_eloss{helper};
    return sample_eloss(rng);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
