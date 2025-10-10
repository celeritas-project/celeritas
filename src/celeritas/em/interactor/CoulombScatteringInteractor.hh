//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/CoulombScatteringInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/CoulombScatteringData.hh"
#include "celeritas/em/data/WentzelOKVIData.hh"
#include "celeritas/em/distribution/WentzelDistribution.hh"
#include "celeritas/em/xs/WentzelHelper.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/InteractionUtils.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "detail/PhysicsConstants.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Applies the Wentzel single Coulomb scattering model.
 *
 * This models incident high-energy electrons and positrons elastically
 * scattering off of nuclei and atomic electrons. Scattering off of the nucleus
 * versus electrons is randomly sampled based on the relative cross-sections
 * (see \c celeritas::WentzelHelper).
 * Production cuts
 * are used to determine the maximum scattering angle off of electrons.
 *
 * This performs the same sampling as in Geant4's \c G4eCoulombScatteringModel,
 * as documented in \cite{g4prm} (release 11.1, section 8.2).
 *
 * \todo When hadronic EM processes are supported, this should be extended to
 * emit secondary ions.
 */
class CoulombScatteringInteractor
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using MomentumSq = units::MevMomentumSq;
    //!@}

  public:
    //! Construct with shared and state data
    inline CELER_FUNCTION
    CoulombScatteringInteractor(CoulombScatteringData const& shared,
                                NativeCRef<WentzelOKVIData> const& wentzel,
                                ParticleTrackView const& particle,
                                Real3 const& inc_direction,
                                MaterialView const& material,
                                IsotopeView const& target,
                                ElementId el_id,
                                CutoffView const& cutoffs);

    //! Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Incident direction
    Real3 const& inc_direction_;

    // Incident particle
    ParticleTrackView const& particle_;

    // Target isotope
    IsotopeView const& target_;

    // Helper for calculating xs ratio and other quantities
    WentzelHelper const helper_;

    // Scattering direction distribution of the Wentzel model
    WentzelDistribution const sample_angle_;

    //// HELPER FUNCTIONS ////

    // Calculates the recoil energy for the given scattering angle
    inline CELER_FUNCTION real_type calc_recoil_energy(real_type cos_theta) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared and state data.
 *
 * \todo Use the proton production cutoff when the recoiled nucleus production
 * is supported.
 */
CELER_FUNCTION
CoulombScatteringInteractor::CoulombScatteringInteractor(
    CoulombScatteringData const& shared,
    NativeCRef<WentzelOKVIData> const& wentzel,
    ParticleTrackView const& particle,
    Real3 const& inc_direction,
    MaterialView const& material,
    IsotopeView const& target,
    ElementId el_id,
    CutoffView const& cutoffs)
    : inc_direction_(inc_direction)
    , particle_(particle)
    , target_(target)
    , helper_(particle,
              material,
              target.atomic_number(),
              wentzel,
              shared.ids,
              cutoffs.energy(shared.ids.electron))
    , sample_angle_(wentzel,
                    helper_,
                    particle,
                    target,
                    el_id,
                    helper_.cos_thetamax_nuclear(),
                    shared.cos_thetamax())
{
    CELER_EXPECT(particle_.particle_id() == shared.ids.electron
                 || particle_.particle_id() == shared.ids.positron);
    CELER_EXPECT(particle_.energy() > zero_quantity()
                 && particle_.energy() < detail::high_energy_limit());
}

//---------------------------------------------------------------------------//
/*!
 * Sample the Coulomb scattering of the incident particle.
 */
template<class Engine>
CELER_FUNCTION Interaction CoulombScatteringInteractor::operator()(Engine& rng)
{
    // Incident particle scatters
    Interaction result;

    // Sample the new direction
    real_type cos_theta = sample_angle_(rng);
    result.direction = ExitingDirectionSampler{cos_theta, inc_direction_}(rng);

    // Recoil energy is kinetic energy transferred to the atom
    real_type inc_energy = value_as<Energy>(particle_.energy());
    real_type recoil_energy = this->calc_recoil_energy(cos_theta);
    CELER_ASSERT(0 <= recoil_energy && recoil_energy <= inc_energy);
    result.energy = Energy{inc_energy - recoil_energy};

    //! \todo For high enough recoil energies, ions are produced
    result.energy_deposition = Energy{recoil_energy};

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculates the recoil energy for the given scattering angle sampled
 * by WentzelDistribution.
 */
CELER_FUNCTION real_type
CoulombScatteringInteractor::calc_recoil_energy(real_type cos_theta) const
{
    real_type projectile_momsq = value_as<MomentumSq>(particle_.momentum_sq());
    real_type projectile_mass = value_as<Mass>(particle_.mass())
                                + value_as<Energy>(particle_.energy());
    real_type target_mass = value_as<Mass>(target_.nuclear_mass());

    return projectile_momsq * (1 - cos_theta)
           / (target_mass + projectile_mass * (1 - cos_theta));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
