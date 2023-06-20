//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/WokviInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/WokviData.hh"
#include "celeritas/em/distribution/WokviDistribution.hh"
#include "celeritas/em/interactor/detail/WokviStateHelper.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Applies the Wentzel OK and VI single Coulomb scattering model.
 */
class WokviInteractor
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION WokviInteractor(WokviRef const& shared,
                                          ParticleTrackView const& particle,
                                          Real3 const& inc_direction,
                                          MaterialView const& material,
                                          ElementComponentId const& elcomp_id,
                                          StackAllocator<Secondary>& allocate);

    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Computed values for this track, shared among multiple parts of the
    // Wokvi model
    const detail::WokviStateHelper state_;

    // Constant shared data
    WokviRef const& data_;

    // Incident direction
    Real3 const& inc_direction_;

    // Allocator for secondary tracks
    StackAllocator<Secondary>& allocate_;

    // Low energy threshold for which particles are absorbed instead of
    // scattered
    const real_type low_energy_threshold_;

    //// HELPER FUNCTIONS ////

    // Calculates the recoil energy for the given scattering direction
    inline CELER_FUNCTION real_type
    calc_recoil_energy(Real3 const& new_direction) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared and state data
 */
CELER_FUNCTION
WokviInteractor::WokviInteractor(WokviRef const& shared,
                                 ParticleTrackView const& particle,
                                 Real3 const& inc_direction,
                                 MaterialView const& material,
                                 ElementComponentId const& elcomp_id,
                                 StackAllocator<Secondary>& allocate)
    : state_(particle,
             material,
             elcomp_id,
             /*cut_energy*/ units::MevEnergy{100.0},
             shared)
    , data_(shared)
    , inc_direction_(inc_direction)
    , allocate_(allocate)
    , low_energy_threshold_(0.01)
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample the Coulomb scattering of the incident particle.
 */
template<class Engine>
CELER_FUNCTION Interaction WokviInteractor::operator()(Engine& rng)
{
    // If below the low energy threshold, the incident particle is absorbed.
    if (state_.inc_energy < low_energy_threshold_)
    {
        Interaction result = Interaction::from_absorption();
        result.energy = Energy{0.0};
        result.energy_deposition = Energy{state_.inc_energy};
        return result;
    }

    // Distribution model governing the scattering
    WokviDistribution distrib(state_, data_);
    if (distrib.cross_section() == 0.0)
    {
        return Interaction::from_unchanged(Energy{state_.inc_energy},
                                           inc_direction_);
    }

    // Incident particle scatters
    Interaction result;

    // Sample the new direction
    const Real3 new_direction = distrib(rng);
    result.direction = rotate(inc_direction_, new_direction);

    // Calculate recoil and final energies
    real_type recoil_energy = calc_recoil_energy(new_direction);
    real_type final_energy = state_.inc_energy - recoil_energy;
    if (final_energy < low_energy_threshold_)
    {
        recoil_energy = state_.inc_energy;
        final_energy = 0.0;
    }
    result.energy = Energy{final_energy};

    // TODO: For high enough recoil energies, ions are produced

    result.energy_deposition = Energy{recoil_energy};

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculates the recoil energy for the given scattering direction calculated
 * by WokviDistribution.
 */
CELER_FUNCTION real_type
WokviInteractor::calc_recoil_energy(Real3 const& new_direction) const
{
    const real_type cos_t_1 = 1.0 - new_direction[2];
    return state_.inc_mom_sq * cos_t_1
           / (state_.target_mass()
              + (state_.inc_mass + state_.inc_energy) * cos_t_1);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
