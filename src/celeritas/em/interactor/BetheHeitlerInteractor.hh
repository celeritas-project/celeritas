//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/BetheHeitlerInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/PolyEvaluator.hh"
#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/BetheHeitlerData.hh"
#include "celeritas/em/distribution/TsaiUrbanDistribution.hh"
#include "celeritas/em/xs/LPMCalculator.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Relativistic model for electron-positron pair production.
 *
 * The energies of the secondary electron and positron are sampled using the
 * Bethe-Heitler cross sections with a Coulomb correction. The LPM effect is
 * taken into account for incident gamma energies above 100 GeV. Exiting
 * particle directions are sampled with the \c TsaiUrbanDistribution . Note
 * that energy is not exactly conserved.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4PairProductionRelModel, as documented in sections 6.5 (gamma conversion)
 * and 10.2.2 (LPM effect) of \cite{g4prm} (release 10.7)
 *
 * For additional context on the derivation see \citet{butcher-electron-1960,
 * https://doi.org/10.1016/0029-5582(60)90162-0} .
 */
class BetheHeitlerInteractor
{
  public:
    //!@{
    //! \name Type aliases
    using Mass = units::MevMass;
    using Energy = units::MevEnergy;
    //!@}

  public:
    //! Construct sampler from shared and state data
    inline CELER_FUNCTION
    BetheHeitlerInteractor(BetheHeitlerData const& shared,
                           ParticleTrackView const& particle,
                           Real3 const& inc_direction,
                           StackAllocator<Secondary>& allocate,
                           MaterialView const& material,
                           ElementView const& element);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// TYPES ////

    using Real2 = Array<real_type, 2>;

    //// DATA ////

    // Shared model data
    BetheHeitlerData const& shared_;
    // Incident gamma energy
    Energy const inc_energy_;
    // Incident direction
    Real3 const& inc_direction_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;
    // Whether LPM supression is applied
    bool const enable_lpm_;
    // Used to calculate the LPM suppression functions
    LPMCalculator calc_lpm_functions_;
    // Minimum epsilon, m_e*c^2/E_gamma; kinematical limit for Y -> e+e-
    real_type epsilon0_;
    // 136/Z^1/3 factor on the screening variable, or zero for low energy
    real_type screen_delta_;
    real_type f_z_;

    //// CONSTANTS ////

    //! Energy below which screening can be neglected
    static CELER_CONSTEXPR_FUNCTION Energy no_screening_threshold()
    {
        return units::MevEnergy{2};
    }

    //! Energy above which the Coulomb correction is applied [MeV]
    static CELER_CONSTEXPR_FUNCTION Energy coulomb_corr_threshold()
    {
        return units::MevEnergy{50};
    }

    //! Energy above which LPM suppression is applied, if enabled [MeV]
    static CELER_CONSTEXPR_FUNCTION Energy lpm_threshold()
    {
        return units::MevEnergy{1e5};
    }

    //// HELPER FUNCTIONS ////

    // Calculate the screening variable \f$ \delta \f$
    inline CELER_FUNCTION real_type impact_parameter(real_type eps) const;

    // Calculate the screening functions \f$ \Phi_1 \f$ and \f$ \Phi_2 \f$
    static inline CELER_FUNCTION Real2 screening_phi(real_type delta);

    // Calculate the auxiliary screening functions \f$ F_1 \f$ and \f$ F_2 \f$
    static inline CELER_FUNCTION Real2 screening_f(real_type delta);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * The incident gamma energy must be at least twice the electron rest mass.
 */
CELER_FUNCTION BetheHeitlerInteractor::BetheHeitlerInteractor(
    BetheHeitlerData const& shared,
    ParticleTrackView const& particle,
    Real3 const& inc_direction,
    StackAllocator<Secondary>& allocate,
    MaterialView const& material,
    ElementView const& element)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
    , enable_lpm_(shared.enable_lpm && inc_energy_ > lpm_threshold())
    , calc_lpm_functions_(
          material, element, shared_.dielectric_suppression(), inc_energy_)
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.gamma);
    CELER_EXPECT(value_as<Energy>(inc_energy_)
                 >= 2 * value_as<Mass>(shared_.electron_mass));

    epsilon0_ = value_as<Mass>(shared_.electron_mass)
                / value_as<Energy>(inc_energy_);

    if (inc_energy_ < no_screening_threshold())
    {
        // Don't reject: just sample uniformly
        screen_delta_ = 0;
        f_z_ = 0;
    }
    else
    {
        screen_delta_ = epsilon0_ * 136 / element.cbrt_z();
        f_z_ = real_type(8) / real_type(3) * element.log_z();
        if (inc_energy_ > coulomb_corr_threshold())
        {
            // Apply Coulomb correction function
            f_z_ += 8 * element.coulomb_correction();
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Sample the distribution.
 */
template<class Engine>
CELER_FUNCTION Interaction BetheHeitlerInteractor::operator()(Engine& rng)
{
    // Allocate space for the electron/positron pair
    Secondary* secondaries = allocate_(2);
    if (secondaries == nullptr)
    {
        // Failed to allocate space for secondaries
        return Interaction::from_failure();
    }

    constexpr real_type half = 0.5;

    // Sample fraction of energy given to electron
    real_type epsilon;
    if (screen_delta_ == 0)
    {
        // If E_gamma < 2 MeV, rejection not needed -- just sample uniformly
        UniformRealDistribution<real_type> sample_eps(epsilon0_, half);
        epsilon = sample_eps(rng);
    }
    else
    {
        // Calculate the minimum (when \epsilon = 1/2) and maximum (when
        // \epsilon = \epsilon_1) values of screening variable, \delta. Above
        // 50 MeV, a Coulomb correction function is introduced.
        real_type const delta_min = 4 * screen_delta_;
        real_type const delta_max
            = std::exp((real_type(42.038) - f_z_) / real_type(8.29))
              - real_type(0.958);
        CELER_ASSERT(delta_min <= delta_max);

        // Calculate the lower limit of epsilon. Due to the Coulomb correction,
        // the cross section can become negative even at kinematically allowed
        // \epsilon > \epsilon_0 values. To exclude these negative cross
        // sections, an additional constraint that \epsilon > \epsilon_1 is
        // introduced, where \epsilon_1 is the solution to
        // \Phi(\delta(\epsilon)) - F(Z)/2 = 0.
        real_type const epsilon1
            = half * (1 - std::sqrt(1 - delta_min / delta_max));
        real_type const epsilon_min = celeritas::max(epsilon0_, epsilon1);

        // Decide to choose f1, g1 [brems] or f2, g2 [pair production]
        // based on N1, N2 (factors from corrected Bethe-Heitler cross section;
        // c.f. Eq. 6.6 of Geant4 Physics Reference 10.6)
        Real2 const fmin = this->screening_f(delta_min) - f_z_;
        BernoulliDistribution choose_f1g1(
            ipow<2>(half - epsilon_min) * fmin[0], real_type(1.5) * fmin[1]);

        // Rejection function g_1 or g_2. Note the it's possible for g to be
        // greater than one
        real_type g;
        do
        {
            if (choose_f1g1(rng))
            {
                // Used to sample from f1
                epsilon = half
                          - (half - epsilon_min)
                                * std::cbrt(generate_canonical(rng));
                CELER_ASSERT(epsilon >= epsilon_min && epsilon <= half);

                // Calculate delta from element atomic number and sampled
                // epsilon
                real_type delta = this->impact_parameter(epsilon);
                CELER_ASSERT(delta <= delta_max && delta >= delta_min);

                // Calculate g_1 rejection function
                if (enable_lpm_)
                {
                    auto screening = screening_phi(delta);
                    auto lpm = calc_lpm_functions_(epsilon);
                    g = lpm.xi
                        * ((2 * lpm.phi + lpm.g) * screening[0]
                           - lpm.g * screening[1] - lpm.phi * f_z_);
                }
                else
                {
                    g = this->screening_f(delta)[0] - f_z_;
                }
                g /= fmin[0];
                CELER_ASSERT(g > 0);
            }
            else
            {
                // Used to sample from f2
                epsilon = UniformRealDistribution{epsilon_min, half}(rng);
                CELER_ASSERT(epsilon >= epsilon_min && epsilon <= half);

                // Calculate delta given the element atomic number and sampled
                // epsilon
                real_type delta = this->impact_parameter(epsilon);
                CELER_ASSERT(delta <= delta_max && delta >= delta_min);

                // Calculate g_2 rejection function
                if (enable_lpm_)
                {
                    auto screening = screening_phi(delta);
                    auto lpm = calc_lpm_functions_(epsilon);
                    g = half * lpm.xi
                        * ((2 * lpm.phi + lpm.g) * screening[0]
                           + lpm.g * screening[1] - (lpm.g + lpm.phi) * f_z_);
                }
                else
                {
                    g = this->screening_f(delta)[1] - f_z_;
                }
                g /= fmin[1];
                CELER_ASSERT(g > 0);
            }
            // TODO: use rejection?
        } while (g < generate_canonical(rng));
    }

    // Construct interaction for change to primary (incident) particle (gamma)
    Interaction result = Interaction::from_absorption();
    result.secondaries = {secondaries, 2};

    // Outgoing secondaries are electron and positron with randomly selected
    // charges
    secondaries[0].particle_id = shared_.ids.electron;
    secondaries[1].particle_id = shared_.ids.positron;

    // Incident energy split between the particles, with rest mass subtracted
    secondaries[0].energy = Energy{(1 - epsilon) * value_as<Energy>(inc_energy_)
                                   - value_as<Mass>(shared_.electron_mass)};
    secondaries[1].energy = Energy{epsilon * value_as<Energy>(inc_energy_)
                                   - value_as<Mass>(shared_.electron_mass)};
    if (BernoulliDistribution(half)(rng))
    {
        trivial_swap(secondaries[0].energy, secondaries[1].energy);
    }

    // Sample secondary directions: note that momentum is not exactly conserved
    real_type const phi = UniformRealDistribution<real_type>(
        0, real_type(2 * constants::pi))(rng);
    auto sample_costheta = [&](Energy e) {
        return TsaiUrbanDistribution{e, shared_.electron_mass}(rng);
    };

    // Note that positron has opposite azimuthal angle
    secondaries[0].direction
        = rotate(from_spherical(sample_costheta(secondaries[0].energy), phi),
                 inc_direction_);
    secondaries[1].direction
        = rotate(from_spherical(sample_costheta(secondaries[1].energy),
                                phi + constants::pi),
                 inc_direction_);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Screening variable \f$ \delta \f$.
 *
 * \f$ \delta \f$ is a function of \f$ \epsilon \f$ and is a measure of the
 * "impact parameter" of the incident photon.
 */
CELER_FUNCTION real_type
BetheHeitlerInteractor::impact_parameter(real_type eps) const
{
    return screen_delta_ / (eps * (1 - eps));
}

//---------------------------------------------------------------------------//
/*!
 * Screening functions \f$ \Phi_1(\delta) \f$ and \f$ \Phi_2(\delta) \f$.
 *
 * These correspond to \em f in Butcher: the first screening function is based
 * on the bremsstrahlung cross sections, and the second is due to pair
 * production. The values are improved function fits by M Novak (Geant4).
 */
CELER_FUNCTION auto
BetheHeitlerInteractor::screening_phi(real_type delta) -> Real2
{
    using R = real_type;

    Real2 result;
    if (delta > R(1.4))
    {
        result[0] = R(21.0190) - R(4.145) * std::log(delta + R(0.958));
        result[1] = result[0];
    }
    else
    {
        using PolyQuad = PolyEvaluator<real_type, 2>;
        result[0] = PolyQuad{20.806, -3.190, 0.5710}(delta);
        result[1] = PolyQuad{20.234, -2.126, 0.0903}(delta);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Auxiliary screening functions \f$ F_1(\delta) \f$ and \f$ F_2(\delta) \f$.
 *
 * The functions \f$ F_1 = 3 \Phi_1(\delta) - \Phi_2(\delta) \f$
 * and \f$ F_2 = 1.5\Phi_1(\delta) + 0.5\Phi_2(\delta) \f$
 * are decreasing functions of \f$ \delta \f$ for all \f$ \delta \f$
 * in \f$ [\delta_\textrm{min}, \delta_\textrm{max}] \f$.
 * They reach their maximum value at
 * \f$ \delta_\textrm{min} = \delta(\epsilon = 1/2)\f$. They are used in the
 * composition + rejection technique for sampling \f$ \epsilon \f$.
 *
 * Note that there's a typo in the Geant4 manual in the formula for F2:
 * subtraction should be addition.
 */
CELER_FUNCTION auto
BetheHeitlerInteractor::screening_f(real_type delta) -> Real2
{
    using R = real_type;
    auto temp = screening_phi(delta);
    return {3 * temp[0] - temp[1], R{1.5} * temp[0] + R{0.5} * temp[1]};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
