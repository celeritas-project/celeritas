//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/BetheHeitlerInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/BetheHeitlerData.hh"
#include "celeritas/em/distribution/TsaiUrbanDistribution.hh"
#include "celeritas/em/xs/LPMCalculator.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Relativistic model for electron-positron pair production.
 *
 * The energies of the secondary electron and positron are sampled using the
 * Bethe-Heitler cross sections with a Coulomb correction. The LPM effect is
 * taken into account for incident gamma energies above 100 GeV.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4PairProductionRelModel, as documented in sections 6.5 (gamma conversion)
 * and 10.2.2 (LPM effect) of the Geant4 Physics Reference Manual (release
 * 10.7)
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

    //! Screening functions \f$ \Phi_1 \f$ and \f$ \Phi_2 \f$
    struct ScreeningFunctions
    {
        real_type phi1;
        real_type phi2;
    };

    //// DATA ////

    // Shared model data
    BetheHeitlerData const& shared_;
    // Incident gamma energy
    const real_type inc_energy_;
    // Incident direction
    Real3 const& inc_direction_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;
    // Element properties for calculating screening functions and variables
    ElementView const& element_;
    // Whether LPM supression is applied
    bool const enable_lpm_;
    // Used to calculate the LPM suppression functions
    LPMCalculator calc_lpm_functions_;
    // Cached minimum epsilon, m_e*c^2/E_gamma; kinematical limit for Y -> e+e-
    real_type epsilon0_;

    //// CONSTANTS ////

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
    inline CELER_FUNCTION ScreeningFunctions
    screening_phi1_phi2(real_type delta) const;

    // Calculate the auxiliary screening function \f$ F_1 \f$
    inline CELER_FUNCTION real_type screening_f1(real_type delta) const;

    // Calculate the auxiliary screening function \f$ F_2 \f$
    inline CELER_FUNCTION real_type screening_f2(real_type delta) const;
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
    , inc_energy_(value_as<Energy>(particle.energy()))
    , inc_direction_(inc_direction)
    , allocate_(allocate)
    , element_(element)
    , enable_lpm_(shared.enable_lpm
                  && inc_energy_ > value_as<Energy>(lpm_threshold()))
    , calc_lpm_functions_(material,
                          element_,
                          shared_.dielectric_suppression(),
                          Energy{inc_energy_})
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.gamma);
    CELER_EXPECT(inc_energy_ >= 2 * value_as<Mass>(shared_.electron_mass));

    epsilon0_ = value_as<Mass>(shared_.electron_mass) / inc_energy_;
}

//---------------------------------------------------------------------------//
/*!
 * Pair production using the Bethe-Heitler model.
 *
 * See section 6.5 of the Geant4 Physics Reference Manual (Release 10.7).
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

    // If E_gamma < 2 MeV, rejection not needed -- just sample uniformly
    real_type epsilon;
    if (inc_energy_ < value_as<Energy>(units::MevEnergy{2}))
    {
        UniformRealDistribution<real_type> sample_eps(epsilon0_, half);
        epsilon = sample_eps(rng);
    }
    else
    {
        // Calculate the minimum (when \epsilon = 1/2) and maximum (when
        // \epsilon = \epsilon_1) values of screening variable, \delta. Above
        // 50 MeV, a Coulomb correction function is introduced.
        const real_type delta_min = 4 * 136 / element_.cbrt_z() * epsilon0_;
        real_type f_z = real_type(8) / real_type(3) * element_.log_z();
        if (inc_energy_ > value_as<Energy>(coulomb_corr_threshold()))
        {
            f_z += 8 * element_.coulomb_correction();
        }
        const real_type delta_max
            = std::exp((real_type(42.038) - f_z) / real_type(8.29))
              - real_type(0.958);
        CELER_ASSERT(delta_min <= delta_max);

        // Calculate the lower limit of epsilon. Due to the Coulomb correction,
        // the cross section can become negative even at kinematically allowed
        // \epsilon > \epsilon_0 values. To exclude these negative cross
        // sections, an additional constraint that \epsilon > \epsilon_1 is
        // introduced, where \epsilon_1 is the solution to
        // \Phi(\delta(\epsilon)) - F(Z)/2 = 0.
        const real_type epsilon1
            = half - half * std::sqrt(1 - delta_min / delta_max);
        const real_type epsilon_min = celeritas::max(epsilon0_, epsilon1);

        // Decide to choose f1, g1 or f2, g2 based on N1, N2 (factors from
        // corrected Bethe-Heitler cross section; c.f. Eq. 6.6 of Geant4
        // Physics Reference 10.6)
        const real_type f10 = this->screening_f1(delta_min) - f_z;
        const real_type f20 = this->screening_f2(delta_min) - f_z;
        BernoulliDistribution choose_f1g1(ipow<2>(half - epsilon_min) * f10,
                                          real_type(1.5) * f20);

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
                    auto screening = screening_phi1_phi2(delta);
                    auto lpm = calc_lpm_functions_(epsilon);
                    g = lpm.xi
                        * ((2 * lpm.phi + lpm.g) * screening.phi1
                           - lpm.g * screening.phi2 - lpm.phi * f_z)
                        / f10;
                }
                else
                {
                    g = (this->screening_f1(delta) - f_z) / f10;
                }
                CELER_ASSERT(g > 0);
            }
            else
            {
                // Used to sample from f2
                epsilon = epsilon_min
                          + (half - epsilon_min) * generate_canonical(rng);
                CELER_ASSERT(epsilon >= epsilon_min && epsilon <= half);

                // Calculate delta given the element atomic number and sampled
                // epsilon
                real_type delta = this->impact_parameter(epsilon);
                CELER_ASSERT(delta <= delta_max && delta >= delta_min);

                // Calculate g_2 rejection function
                if (enable_lpm_)
                {
                    auto screening = screening_phi1_phi2(delta);
                    auto lpm = calc_lpm_functions_(epsilon);
                    g = lpm.xi
                        * ((lpm.phi + half * lpm.g) * screening.phi1
                           + half * lpm.g * screening.phi2
                           - half * (lpm.g + lpm.phi) * f_z)
                        / f20;
                }
                else
                {
                    g = (this->screening_f2(delta) - f_z) / f20;
                }
                CELER_ASSERT(g > 0);
            }
        } while (g < generate_canonical(rng));
    }

    // Construct interaction for change to primary (incident) particle (gamma)
    Interaction result = Interaction::from_absorption();
    result.secondaries = {secondaries, 2};

    // Outgoing secondaries are electron and positron
    secondaries[0].particle_id = shared_.ids.electron;
    secondaries[1].particle_id = shared_.ids.positron;

    secondaries[0].energy = Energy{(1 - epsilon) * inc_energy_
                                   - value_as<Mass>(shared_.electron_mass)};
    secondaries[1].energy = Energy{epsilon * inc_energy_
                                   - value_as<Mass>(shared_.electron_mass)};

    // Select charges for child particles (e-, e+) randomly
    if (BernoulliDistribution(half)(rng))
    {
        trivial_swap(secondaries[0].energy, secondaries[1].energy);
    }

    // Sample secondary directions.
    // Note that momentum is not exactly conserved.
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
    real_type phi = sample_phi(rng);

    // Electron
    TsaiUrbanDistribution sample_electron_angle(secondaries[0].energy,
                                                shared_.electron_mass);
    real_type cost = sample_electron_angle(rng);
    secondaries[0].direction
        = rotate(from_spherical(cost, phi), inc_direction_);
    // Positron
    TsaiUrbanDistribution sample_positron_angle(secondaries[1].energy,
                                                shared_.electron_mass);
    cost = sample_positron_angle(rng);
    secondaries[1].direction
        = rotate(from_spherical(cost, phi + constants::pi), inc_direction_);
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
    return 136 / element_.cbrt_z() * epsilon0_ / (eps * (1 - eps));
}

//---------------------------------------------------------------------------//
/*!
 * Screening functions \f$ \Phi_1(\delta) \f$ and \f$ \Phi_2(\delta) \f$.
 */
CELER_FUNCTION auto
BetheHeitlerInteractor::screening_phi1_phi2(real_type delta) const
    -> ScreeningFunctions
{
    using R = real_type;

    ScreeningFunctions result;
    if (delta > R(1.4))
    {
        result.phi1 = R(21.0190) - R(4.145) * std::log(delta + R(0.958));
        result.phi2 = result.phi1;
    }
    else
    {
        result.phi1 = R(20.806) - delta * (R(3.190) - R(0.5710) * delta);
        result.phi2 = R(20.234) - delta * (R(2.126) - R(0.0903) * delta);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Auxiliary screening functions \f$ F_1(\delta) \f$ and \f$ F_2(\delta) \f$.
 *
 * The functions \f$ F_1 = 3\Phi_1(\delta) - \Phi_2(\delta) \f$ and \f$ F_2 =
 * 1.5\Phi_1(\delta) - 0.5\Phi_2(\delta) \f$ are decreasing functions of \f$
 * \delta \f$ for all \f$ \delta \f$ in \f$ [\delta_\textrm{min},
 * \delta_\textrm{max}] \f$. They reach their maximum value at \f$
 * \delta_\textrm{min} = \delta(\epsilon = 1/2)\f$. They are used in the
 * composition + rejection technique for sampling \f$ \epsilon \f$.
 */
CELER_FUNCTION real_type BetheHeitlerInteractor::screening_f1(real_type delta) const
{
    using R = real_type;

    return delta > R(1.4) ? R(42.038) - R(8.29) * std::log(delta + R(0.958))
                          : R(42.184) - delta * (R(7.444) - R(1.623) * delta);
}

CELER_FUNCTION real_type BetheHeitlerInteractor::screening_f2(real_type delta) const
{
    using R = real_type;

    return delta > R(1.4) ? R(42.038) - R(8.29) * std::log(delta + R(0.958))
                          : R(41.326) - delta * (R(5.848) - R(0.902) * delta);
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
