//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/UrbanMscHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/grid/EnergyLossCalculator.hh"
#include "celeritas/grid/InverseRangeCalculator.hh"
#include "celeritas/grid/RangeCalculator.hh"
#include "celeritas/Types.hh"

#include "celeritas/em/data/UrbanMscData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * This is a helper class for the UrbanMscStepLimit and UrbanMscScatter.
 */
class UrbanMscHelper
{
  public:
    //!@{
    //! Type aliases
    using Energy       = units::MevEnergy;
    using MaterialData = detail::UrbanMscMaterialData;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION UrbanMscHelper(const UrbanMscRef&       shared,
                                         const ParticleTrackView& particle,
                                         const PhysicsTrackView&  physics);

    //// HELPER FUNCTIONS ////

    //! The slowing-down range for the starting particle energy
    CELER_FUNCTION real_type range() const { return range_; }

    // The mean free path of the multiple scattering for a given energy
    inline CELER_FUNCTION real_type msc_mfp(Energy energy) const;

    // TODO: the following two methods are used only by MscStepLimit

    // The total energy loss over a given step length
    inline CELER_FUNCTION Energy calc_eloss(real_type step) const;

    // The kinetic energy at the end of a given step length corrected by dedx
    inline CELER_FUNCTION Energy calc_end_energy(real_type step) const;

  private:
    //// DATA ////

    // Incident particle energy
    const Energy inc_energy_;
    // PhysicsTrackView
    const PhysicsTrackView& physics_;
    // Range scaling factor
    const real_type dtrl_;

    // Shared value of range
    real_type range_;
    // Process ID of the eletromagnetic_msc process
    ParticleProcessId msc_pid_;
    // Grid ID of range value of the energy loss
    ValueGridId range_gid_;
    // Grid ID of dedx value of the energy loss
    ValueGridId eloss_gid_;
    // Grid ID of the lambda value of MSC
    ValueGridId mfp_gid_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
UrbanMscHelper::UrbanMscHelper(const UrbanMscRef&       shared,
                               const ParticleTrackView& particle,
                               const PhysicsTrackView&  physics)
    : inc_energy_(particle.energy())
    , physics_(physics)
    , dtrl_(shared.params.dtrl())
{
    CELER_EXPECT(particle.particle_id() == shared.ids.electron
                 || particle.particle_id() == shared.ids.positron);

    ParticleProcessId eloss_pid = physics.eloss_ppid();
    range_gid_ = physics.value_grid(ValueGridType::range, eloss_pid);
    eloss_gid_ = physics.value_grid(ValueGridType::energy_loss, eloss_pid);
    msc_pid_   = physics_.msc_ppid();
    mfp_gid_   = physics_.value_grid(ValueGridType::msc_mfp, msc_pid_);
    range_ = physics.make_calculator<RangeCalculator>(range_gid_)(inc_energy_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the mean free path of the msc for a given particle energy.
 */
CELER_FUNCTION real_type UrbanMscHelper::msc_mfp(Energy energy) const
{
    real_type xsec = physics_.calc_xs(msc_pid_, mfp_gid_, energy)
                     / ipow<2>(energy.value());
    CELER_ENSURE(xsec >= 0);
    return 1 / xsec;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the total energy loss over a given step length.
 */
CELER_FUNCTION auto UrbanMscHelper::calc_eloss(real_type step) const -> Energy
{
    auto calc_energy
        = physics_.make_calculator<InverseRangeCalculator>(range_gid_);

    return calc_energy(step);
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the kinetic energy at the end of a given msc step.
 */
CELER_FUNCTION auto UrbanMscHelper::calc_end_energy(real_type step) const
    -> Energy
{
    if (step <= range_ * dtrl_)
    {
        // Short step can be approximated with linear extrapolation.
        real_type dedx = physics_.make_calculator<EnergyLossCalculator>(
            eloss_gid_)(inc_energy_);

        return Energy{inc_energy_.value() - step * dedx};
    }
    else
    {
        // Longer step is calculated exactly with inverse range
        return this->calc_eloss(range_ - step);
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
