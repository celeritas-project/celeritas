//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/WentzelRatioCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/ParticleTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the ratio of the electron to total Wentzel cross sections for
 * elastic Coulomb scattering.
 */
class WentzelRatioCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using MomentumSq = units::MevMomentumSq;
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    //!@}

  public:
    //! Construct the calculator from the given values
    inline CELER_FUNCTION
    WentzelRatioCalculator(ParticleTrackView const& particle,
                           AtomicNumber target_z,
                           WentzelRef const& data,
                           real_type cutoff_energy);

    // The ratio of electron to total cross section for Coulomb scattering
    inline CELER_FUNCTION real_type operator()() const;

    // Moilere screening coefficient
    inline CELER_FUNCTION real_type screening_coefficient() const;

    // (Cosine of) the maximum scattering angle off of electrons
    inline CELER_FUNCTION real_type cos_t_max_elec() const;

  private:
    // Target atomic number
    AtomicNumber const target_z_;

    // Moliere screening coefficient
    real_type screening_coefficient_;

    // Cosine of the maximum scattering angle off of electrons
    real_type cos_t_max_elec_;

    // Shared WentzelModel data
    WentzelRef const& data_;

    //! Calculate the Moilere screening coefficient
    inline CELER_FUNCTION real_type
    calc_screening_coefficient(ParticleTrackView const& particle) const;

    //! Calculate the screening coefficient R^2 for electrons
    CELER_CONSTEXPR_FUNCTION real_type screen_r_sq_elec() const;

    //! Calculate the (cosine of) the maximum scattering angle off of electrons
    inline CELER_FUNCTION real_type calc_max_electron_cos_t(
        ParticleTrackView const& particle, real_type cutoff_energy) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with state data.
 */
CELER_FUNCTION
WentzelRatioCalculator::WentzelRatioCalculator(ParticleTrackView const& particle,
                                               AtomicNumber target_z,
                                               WentzelRef const& data,
                                               real_type cutoff_energy)
    : target_z_(target_z), data_(data)
{
    screening_coefficient_ = calc_screening_coefficient(particle)
                             * data_.screening_factor;
    cos_t_max_elec_ = calc_max_electron_cos_t(particle, cutoff_energy);

    CELER_EXPECT(target_z_.get() > 0);
    CELER_EXPECT(screening_coefficient_ > 0);
    CELER_EXPECT(cos_t_max_elec_ >= -1 && cos_t_max_elec_ <= 1);
}

//---------------------------------------------------------------------------//
/*!
 * Ratio of electron cross section to the total (nuclear + electron)
 * cross section.
 */
CELER_FUNCTION real_type WentzelRatioCalculator::operator()() const
{
    // Calculating only reduced cross sections by elimination mutual factors
    // in the ratio.
    real_type nuc_xsec = target_z_.get() / (1 + screening_coefficient_);
    real_type elec_xsec = (1 - cos_t_max_elec_)
                          / (1 - cos_t_max_elec_ + 2 * screening_coefficient_);

    return elec_xsec / (nuc_xsec + elec_xsec);
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the cached Moilere screening coefficient.
 */
CELER_FUNCTION real_type WentzelRatioCalculator::screening_coefficient() const
{
    return screening_coefficient_;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the Moilere screening coefficient as in [PRM] eqn 8.51.
 */
CELER_FUNCTION real_type WentzelRatioCalculator::calc_screening_coefficient(
    ParticleTrackView const& particle) const
{
    // TODO: Reference for just proton correction?
    real_type correction = 1;
    real_type sq_cbrt_z = fastpow(real_type(target_z_.get()), real_type{2} / 3);
    if (target_z_.get() > 1)
    {
        real_type tau = value_as<Energy>(particle.energy())
                        / value_as<Mass>(particle.mass());
        // TODO: Reference for this factor?
        real_type factor = std::sqrt(tau / (tau + sq_cbrt_z));

        correction = min(target_z_.get() * real_type{1.13},
                         real_type{1.13}
                             + real_type{3.76}
                                   * ipow<2>(target_z_.get()
                                             * constants::alpha_fine_structure)
                                   * factor / particle.beta_sq());
    }

    return correction * screen_r_sq_elec() * sq_cbrt_z
           / value_as<MomentumSq>(particle.momentum_sq());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the screening R^2 coefficient for incident electrons. This is
 * the constant prefactor of [PRM] eqn 8.51.
 */
CELER_CONSTEXPR_FUNCTION real_type WentzelRatioCalculator::screen_r_sq_elec() const
{
    //! Thomas-Fermi constant C_TF
    //! \f$ \frac{1}{2}\left(\frac{3\pi}{4}\right)^{2/3} \f$
    constexpr real_type ctf = 0.8853413770001135;

    return native_value_to<MomentumSq>(
               ipow<2>(constants::hbar_planck / (2 * ctf * constants::a0_bohr)))
        .value();
}

//---------------------------------------------------------------------------//
/*!
 * (Cosine of) the maximum polar angle that incident particles scatter off
 * of the target's electrons.
 */
CELER_FUNCTION real_type WentzelRatioCalculator::cos_t_max_elec() const
{
    return cos_t_max_elec_;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the (cosine of the) maximum polar angle that incident particle
 * can scatter off of the target's electrons.
 */
CELER_FUNCTION real_type WentzelRatioCalculator::calc_max_electron_cos_t(
    ParticleTrackView const& particle, real_type cutoff_energy) const
{
    real_type inc_energy = value_as<Energy>(particle.energy());
    real_type mass = value_as<Mass>(particle.mass());

    real_type max_energy = (particle.particle_id() == data_.ids.electron)
                               ? real_type{0.5} * inc_energy
                               : inc_energy;
    real_type final_energy = inc_energy - min(cutoff_energy, max_energy);

    if (final_energy > 0)
    {
        real_type incident_ratio = 1 + 2 * mass / inc_energy;
        real_type final_ratio = 1 + 2 * mass / final_energy;
        real_type cos_t_max = std::sqrt(incident_ratio / final_ratio);

        return clamp(cos_t_max, real_type{0}, real_type{1});
    }
    else
    {
        return 0;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
