//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ScintillationGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/ExponentialDistribution.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"
#include "celeritas/random/distribution/NormalDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "OpticalPrimary.hh"
#include "ScintillationData.hh"
#include "ScintillationInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample Scintillation photons.
 *
 * \note This performs the same sampling routine as in G4Scintillation class
 * of the Geant4 release 11.2 with some modifications.
 */
class ScintillationGenerator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct from scintillation properties and input data
    inline CELER_FUNCTION
    ScintillationGenerator(ScintillationInput const& input,
                           NativeCRef<ScintillationData> const& shared,
                           Span<OpticalPrimary> photons);

    // Sample Scintillation photons from input data
    template<class Generator>
    inline CELER_FUNCTION Span<OpticalPrimary> operator()(Generator& rng);

    // TODO: Move it to detail/OpticalUtils.
    static CELER_CONSTEXPR_FUNCTION real_type hc()
    {
        return constants::h_planck * constants::c_light / units::Mev::value();
    }

  private:
    //// TYPES ////

    using UniformRealDist = UniformRealDistribution<real_type>;
    using ExponentialDist = ExponentialDistribution<real_type>;

    //// DATA ////

    ScintillationInput const& input_;
    ScintillationSpectra const& spectra_;
    Span<OpticalPrimary> photons_;

    UniformRealDist sample_cost_;
    UniformRealDist sample_phi_;
    ExponentialDist sample_time_;

    bool is_neutral_{};
    real_type delta_velocity_{};
    Real3 delta_pos_{};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared scintillation data and input.
 */
CELER_FUNCTION
ScintillationGenerator::ScintillationGenerator(
    ScintillationInput const& input,
    NativeCRef<ScintillationData> const& shared,
    Span<OpticalPrimary> photons)
    : input_(input)
    , spectra_(shared.spectra[input.matId])
    , photons_(photons)
    , sample_cost_(-1, 1)
    , sample_phi_(0, 2 * constants::pi)
    , sample_time_(real_type{1} / spectra_.fall_time[0])
    , is_neutral_{native_value_from(input_.charge) == 0}
{
    CELER_EXPECT(input_);
    CELER_EXPECT(photons_.size() == input_.num_photons);

    delta_pos_ = input_.post_pos - input_.pre_pos;
    delta_velocity_ = input_.post_velocity - input_.pre_velocity;
}

//---------------------------------------------------------------------------//
/*!
 * Sample scintillation photons from optical property data and step data.
 *
 * The optical photons are generated evenly along the step and are emitted
 * uniformly over the entire solid angle with a random linear polarization.
 * The photon energy is calculated by the scintillation emission wavelength
 * \f[
   E = \frac{hc}{\lambda},
 * \f]
 * where \f$ h \f$ is the Planck constant and \f$ c \f$ is the speed of light,
 * and \f$ \lambda \f$ is sampled by the normal distribution with the mean of
 * scintillation emission spectra and the standard deviation. The emitted time
 * is simulated according to empirical shapes of the material-dependent
 * scintillation time structure with one or double exponentials.
 */
template<class Generator>
CELER_FUNCTION Span<OpticalPrimary>
ScintillationGenerator::operator()(Generator& rng)
{
    // Loop for generating scintillation photons
    size_type num_generated{0};

    for (auto sid : range(3))
    {
        NormalDistribution<real_type> sample_lambda_(
            spectra_.lambda_mean[sid], spectra_.lambda_sigma[sid]);

        size_type num_photons = input_.num_photons * spectra_.yield_prob[sid];

        CELER_EXPECT(num_generated + num_photons <= input_.num_photons);

        for (size_type i : range(num_generated, num_generated + num_photons))
        {
            // Sample wavelength and convert to energy
            real_type wave_length = sample_lambda_(rng);
            CELER_EXPECT(wave_length > 0);
            photons_[i].energy = Energy(this->hc() / wave_length);

            // Sample direction
            real_type cost = sample_cost_(rng);
            real_type phi = sample_phi_(rng);
            photons_[i].direction
                = rotate(from_spherical(cost, phi), input_.post_pos);

            // Sample polarization
            photons_[i].polarization
                = from_spherical(-std::sqrt(1 - ipow<2>(cost)), phi);
            Real3 perp = cross_product(photons_[i].polarization,
                                       photons_[i].direction);
            phi = sample_phi_(rng);
            photons_[i].polarization = std::cos(phi) * photons_[i].polarization
                                       + std::sin(phi) * perp;

            // Sample position
            real_type u = (is_neutral_) ? 1 : generate_canonical(rng);
            photons_[i].position = input_.pre_pos + u * delta_pos_;

            // Sample time
            real_type delta_time = u * input_.step_length
                                   / (input_.pre_velocity
                                      + u * real_type(0.5) * delta_velocity_);

            if (spectra_.rise_time[sid] == 0)
            {
                delta_time -= spectra_.fall_time[sid]
                              * std::log(generate_canonical(rng));
            }
            else
            {
                real_type scint_time{};
                real_type envelop{};
                do
                {
                    scint_time = sample_time_(rng);
                    envelop
                        = 1 - std::exp(-scint_time / spectra_.rise_time[sid]);
                } while (!BernoulliDistribution(envelop)(rng));
                delta_time += scint_time;
            }
            photons_[i].time = input_.time + delta_time;
        }
        num_generated += num_photons;
    }

    return photons_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
