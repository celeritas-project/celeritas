//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishinaInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/ArrayUtils.hh"
#include "base/Constants.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "base/StackAllocator.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "random/distributions/BernoulliDistribution.hh"
#include "random/distributions/ReciprocalDistribution.hh"
#include "random/distributions/UniformRealDistribution.hh"
#include "KleinNishinaData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Perform Compton scattering, neglecting atomic binding energy.
 *
 * This is a model for the discrete Compton inelastic scattering process. Given
 * an incident gamma, it produces a single secondary (electron)
 * and returns an interaction for the change to the incident gamma
 * direction and energy. No cutoffs are performed for the incident energy or
 * the exiting gamma energy. A secondary production cutoff is applied to the
 * outgoing electron.
 *
 * \note This performs the same sampling routine as in Geant4's
 *  G4KleinNishinaCompton, as documented in section 6.4.2 of the Geant4 Physics
 *  Reference (release 10.6).
 */
class KleinNishinaInteractor
{
  public:
    // Construct from shared and state data
    inline CELER_FUNCTION
    KleinNishinaInteractor(const KleinNishinaData&  shared,
                           const ParticleTrackView& particle,
                           const Real3&             inc_direction);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

    //! Energy threshold for secondary production [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy secondary_cutoff()
    {
        return units::MevEnergy{1e-4};
    }

  private:
    // Constant data
    const KleinNishinaData& shared_;
    // Incident gamma energy
    const units::MevEnergy inc_energy_;
    // Incident direction
    const Real3& inc_direction_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * The incident particle must be above the energy threshold: this should be
 * handled in code *before* the interactor is constructed.
 */
CELER_FUNCTION
KleinNishinaInteractor::KleinNishinaInteractor(const KleinNishinaData& shared,
                                               const ParticleTrackView& particle,
                                               const Real3& inc_direction)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , inc_direction_(inc_direction)
{
    CELER_EXPECT(particle.particle_id() == shared_.gamma_id);
    CELER_EXPECT(inc_energy_ > zero_quantity());
}

//---------------------------------------------------------------------------//
/*!
 * Sample Compton scattering using the Klein-Nishina model.
 *
 * See section 6.4.2 of the Geant physics reference. Epsilon is the ratio of
 * outgoing to incident gamma energy, bounded in [epsilon_0, 1].
 */
template<class Engine>
CELER_FUNCTION Interaction KleinNishinaInteractor::operator()(Engine& rng)
{
    using Energy = units::MevEnergy;

    // Value of epsilon corresponding to minimum photon energy
    const real_type inc_energy_per_mecsq = value_as<Energy>(inc_energy_)
                                           * shared_.inv_electron_mass;
    const real_type epsilon_0 = 1 / (1 + 2 * inc_energy_per_mecsq);

    // Probability of alpha_1 to choose f_1 (sample epsilon)
    BernoulliDistribution choose_f1(-std::log(epsilon_0),
                                    real_type(0.5) * (1 - ipow<2>(epsilon_0)));
    // Sample f_1(\eps) \propto 1/\eps on [\eps_0, 1]
    ReciprocalDistribution<real_type> sample_f1(epsilon_0);
    // Sample square of f_2(\eps^2) \propto 1 on [\eps_0^2, 1]
    UniformRealDistribution<real_type> sample_f2_sq(ipow<2>(epsilon_0), 1);

    // Rejection loop: sample epsilon (energy change) and direction change
    real_type epsilon;
    real_type one_minus_costheta;
    // Temporary sample values used in rejection
    real_type acceptance_prob;
    do
    {
        // Sample epsilon and square
        real_type epsilon_sq;
        if (choose_f1(rng))
        {
            // Sample f_1(\eps) \propto 1/\eps on [\eps_0, 1]
            // => \eps \gets \eps_0^\xi = \exp(\xi \log \eps_0)
            epsilon    = sample_f1(rng);
            epsilon_sq = epsilon * epsilon;
        }
        else
        {
            // Sample f_2(\eps) = 2 * \eps / (1 - epsilon_0 * epsilon_0)
            epsilon_sq = sample_f2_sq(rng);
            epsilon    = std::sqrt(epsilon_sq);
        }
        CELER_ASSERT(epsilon >= epsilon_0 && epsilon <= 1);

        // Calculate angles: need sin^2 \theta for rejection
        one_minus_costheta = (1 - epsilon) / (epsilon * inc_energy_per_mecsq);
        CELER_ASSERT(one_minus_costheta >= 0 && one_minus_costheta <= 2);
        real_type sintheta_sq = one_minus_costheta * (2 - one_minus_costheta);
        acceptance_prob       = epsilon * sintheta_sq / (1 + epsilon_sq);
    } while (BernoulliDistribution(acceptance_prob)(rng));

    // Construct interaction for change to primary (incident) particle
    Interaction result;
    result.action    = Action::scattered;
    result.energy    = Energy{epsilon * inc_energy_.value()};
    result.direction = inc_direction_;

    // Sample azimuthal direction and rotate the outgoing direction
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
    result.direction
        = rotate(from_spherical(1 - one_minus_costheta, sample_phi(rng)),
                 result.direction);

    // Construct secondary energy by neglecting electron binding energy
    result.secondary.energy
        = Energy{inc_energy_.value() - result.energy.value()};

    // Apply secondary production cutoff
    if (result.secondary.energy < KleinNishinaInteractor::secondary_cutoff())
    {
        result.energy_deposition = result.secondary.energy;
        result.secondary         = {};
        return result;
    }

    // Outgoing secondary is an electron
    result.secondary.particle_id = shared_.electron_id;
    // Calculate exiting electron direction via conservation of momentum
    for (int i = 0; i < 3; ++i)
    {
        result.secondary.direction[i] = inc_direction_[i] * inc_energy_.value()
                                        - result.direction[i]
                                              * result.energy.value();
    }
    normalize_direction(&result.secondary.direction);

    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
