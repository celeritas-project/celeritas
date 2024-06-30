//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/MuBetheBlochInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuBetheBlochData.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "detail/PhysicsConstants.hh"
#include "detail/Utils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform the discrete part of the muon ionization process.
 *
 * This simulates the production of delta rays by incident mu- and mu+ with
 * energies above 200 keV.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4MuBetheBlochModel and as documented in the Geant4 Physics Reference Manual
 * (Release 11.1) section 11.1.
 */
class MuBetheBlochInteractor
{
  public:
    //!@{
    //! \name Type aliases
    using Mass = units::MevMass;
    using Energy = units::MevEnergy;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    //! Construct with shared and state data
    inline CELER_FUNCTION
    MuBetheBlochInteractor(MuBetheBlochData const& shared,
                           ParticleTrackView const& particle,
                           CutoffView const& cutoffs,
                           Real3 const& inc_direction,
                           StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Shared constant physics properties
    MuBetheBlochData const& shared_;
    // Allocate space for the secondary particle
    StackAllocator<Secondary>& allocate_;
    // Incident direction
    Real3 const& inc_direction_;
    // Incident particle energy [MeV]
    real_type inc_energy_;
    // Incident particle momentum [MeV / c]
    real_type inc_momentum_;
    // Square of fractional speed of light
    real_type beta_sq_;
    // Muon mass
    real_type mass_;
    // Electron mass
    real_type electron_mass_;
    // Total energy of the incident particle [MeV]
    real_type total_energy_;
    // Secondary electron cutoff for current material [MeV]
    real_type electron_cutoff_;
    // Maximum energy of the secondary electron [MeV]
    real_type max_secondary_energy_;
    // Whether to apply the radiative correction
    bool use_rad_correction_;
    // Envelope distribution for rejection sampling
    real_type envelope_dist_;

    //// CONSTANTS ////

    //! Incident energy above which the radiative correction is applied [MeV]
    static CELER_CONSTEXPR_FUNCTION Energy rad_correction_limit()
    {
        return units::MevEnergy{250};  //!< 250 MeV
    }

    //! Lower limit of radiative correction integral [MeV]
    static CELER_CONSTEXPR_FUNCTION Energy kin_energy_limit()
    {
        // This is the lower limit of the energy transfer from the incident
        // muon to the delta ray and radiative gammas
        return units::MevEnergy{0.1};  //!< 100 keV
    }

    //! Fine structure constant over two pi
    static CELER_CONSTEXPR_FUNCTION real_type alpha_over_twopi()
    {
        return constants::alpha_fine_structure / (2 * constants::pi);
    }

    //// HELPER FUNCTIONS ////

    inline CELER_FUNCTION real_type calc_max_secondary_energy() const;
    inline CELER_FUNCTION real_type calc_envelope_distribution() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION MuBetheBlochInteractor::MuBetheBlochInteractor(
    MuBetheBlochData const& shared,
    ParticleTrackView const& particle,
    CutoffView const& cutoffs,
    Real3 const& inc_direction,
    StackAllocator<Secondary>& allocate)
    : shared_(shared)
    , allocate_(allocate)
    , inc_direction_(inc_direction)
    , inc_energy_(value_as<Energy>(particle.energy()))
    , inc_momentum_(value_as<Momentum>(particle.momentum()))
    , beta_sq_(particle.beta_sq())
    , mass_(value_as<Mass>(particle.mass()))
    , electron_mass_(value_as<Mass>(shared_.electron_mass))
    , total_energy_(inc_energy_ + mass_)
    , electron_cutoff_(value_as<Energy>(cutoffs.energy(shared_.electron)))
    , max_secondary_energy_(this->calc_max_secondary_energy())
    , use_rad_correction_(
          inc_energy_ > value_as<Energy>(rad_correction_limit())
          && max_secondary_energy_ > value_as<Energy>(kin_energy_limit()))
    , envelope_dist_(this->calc_envelope_distribution())
{
    CELER_EXPECT(particle.particle_id() == shared_.mu_minus
                 || particle.particle_id() == shared_.mu_plus);
    CELER_EXPECT(inc_energy_ > electron_cutoff_);
    CELER_EXPECT(inc_energy_
                 >= value_as<Energy>(detail::mu_bethe_bloch_lower_limit()));
}

//---------------------------------------------------------------------------//
/*!
 * Simulate discrete muon ionization.
 */
template<class Engine>
CELER_FUNCTION Interaction MuBetheBlochInteractor::operator()(Engine& rng)
{
    if (electron_cutoff_ > max_secondary_energy_)
    {
        // No interaction if the maximum secondary energy is below the cutoff
        return Interaction::from_unchanged();
    }

    // Allocate secondary electron
    Secondary* secondary = allocate_(1);
    if (secondary == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    real_type secondary_energy;
    real_type target_dist;
    do
    {
        secondary_energy = electron_cutoff_ * max_secondary_energy_
                           / UniformRealDistribution(
                               electron_cutoff_, max_secondary_energy_)(rng);
        target_dist = 1 - beta_sq_ * secondary_energy / max_secondary_energy_
                      + ipow<2>(secondary_energy)
                            / (2 * ipow<2>(total_energy_));

        if (use_rad_correction_
            && secondary_energy > value_as<Energy>(kin_energy_limit()))
        {
            real_type a1 = std::log(1 + 2 * secondary_energy / electron_mass_);
            real_type a3 = std::log(4 * total_energy_
                                    * (total_energy_ - secondary_energy)
                                    / ipow<2>(mass_));
            target_dist *= (1 + alpha_over_twopi() * a1 * (a3 - a1));
        }
    } while (!BernoulliDistribution(target_dist / envelope_dist_)(rng));

    // TODO: If the \c UseAngularGeneratorFlag is set (false by default), use
    // the angular generator interface to sample the secondary direction (see
    // \c G4EmParameters::UseAngularGeneratorForIonisation())
    real_type secondary_momentum = std::sqrt(
        secondary_energy * (secondary_energy + 2 * electron_mass_));
    real_type total_momentum = total_energy_ * std::sqrt(beta_sq_);
    real_type costheta = secondary_energy * (total_energy_ + electron_mass_)
                         / (secondary_momentum * total_momentum);
    CELER_ASSERT(costheta <= 1);

    // Sample and save outgoing secondary data
    secondary->direction
        = detail::CartesianTransformSampler{costheta, inc_direction_}(rng);
    secondary->energy = Energy{secondary_energy};
    secondary->particle_id = shared_.electron;

    Interaction result;
    result.energy = Energy{inc_energy_ - secondary_energy};
    result.direction = detail::calc_exiting_direction(
        inc_momentum_, secondary_momentum, inc_direction_, secondary->direction);
    result.secondaries = {secondary, 1};

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate maximum kinetic energy of the secondary electron.
 */
CELER_FUNCTION real_type MuBetheBlochInteractor::calc_max_secondary_energy() const
{
    real_type mass_ratio = electron_mass_ / mass_;
    real_type tau = inc_energy_ / mass_;
    return 2 * electron_mass_ * tau * (tau + 2)
           / (1 + 2 * (tau + 1) * mass_ratio + ipow<2>(mass_ratio));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate envelope distribution for rejection sampling of secondary energy.
 */
CELER_FUNCTION real_type MuBetheBlochInteractor::calc_envelope_distribution() const
{
    if (use_rad_correction_)
    {
        return 1
               + alpha_over_twopi()
                     * ipow<2>(std::log(2 * total_energy_ / mass_));
    }
    return 1;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
