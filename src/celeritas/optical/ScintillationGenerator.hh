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

#include "OpticalDistributionData.hh"
#include "OpticalPrimary.hh"
#include "ScintillationData.hh"

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
    // Construct from scintillation data and distribution parameters
    inline CELER_FUNCTION
    ScintillationGenerator(OpticalDistributionData const& dist,
                           NativeCRef<ScintillationData> const& shared,
                           Span<OpticalPrimary> photons);

    // Sample Scintillation photons from the distribution
    template<class Generator>
    inline CELER_FUNCTION Span<OpticalPrimary> operator()(Generator& rng);

  private:
    //// TYPES ////

    using UniformRealDist = UniformRealDistribution<real_type>;
    using ExponentialDist = ExponentialDistribution<real_type>;

    //// DATA ////

    OpticalDistributionData const& dist_;
    NativeCRef<ScintillationData> const& shared_;
    Span<OpticalPrimary> photons_;

    UniformRealDist sample_cost_;
    UniformRealDist sample_phi_;

    bool is_neutral_{};
    units::LightSpeed delta_speed_{};
    Real3 delta_pos_{};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared scintillation data and distribution parameters.
 */
CELER_FUNCTION
ScintillationGenerator::ScintillationGenerator(
    OpticalDistributionData const& dist,
    NativeCRef<ScintillationData> const& shared,
    Span<OpticalPrimary> photons)
    : dist_(dist)
    , shared_(shared)
    , photons_(photons)
    , sample_cost_(-1, 1)
    , sample_phi_(0, 2 * constants::pi)
    , is_neutral_{dist_.charge == zero_quantity()}
{
    CELER_EXPECT(dist_);
    CELER_EXPECT(shared_);
    CELER_EXPECT(photons_.size() == dist_.num_photons);

    auto const& pre_step = dist_.points[StepPoint::pre];
    auto const& post_step = dist_.points[StepPoint::post];
    delta_pos_ = post_step.pos - pre_step.pos;
    delta_speed_ = post_step.speed - pre_step.speed;
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
 * scintillation emission spectrum and the standard deviation. The emitted time
 * is simulated according to empirical shapes of the material-dependent
 * scintillation time structure with one or double exponentials.
 */
template<class Generator>
CELER_FUNCTION Span<OpticalPrimary>
ScintillationGenerator::operator()(Generator& rng)
{
    // Loop for generating scintillation photons
    size_type num_generated{0};

    ScintillationSpectrum const& spectrum = shared_.spectra[dist_.material];

    for (auto sid : spectrum.components)
    {
        ScintillationComponent component = shared_.components[sid];

        // Calculate the number of photons to generate for this component
        size_type num_photons
            = sid.get() + 1 == spectrum.components.size()
                  ? dist_.num_photons - num_generated
                  : static_cast<size_type>(dist_.num_photons
                                           * component.yield_prob);

        CELER_EXPECT(num_generated + num_photons <= dist_.num_photons);

        // Sample photons for each scintillation component
        NormalDistribution<real_type> sample_lambda(component.lambda_mean,
                                                    component.lambda_sigma);
        ExponentialDist sample_time(real_type{1} / component.fall_time);

        for (size_type i : range(num_generated, num_generated + num_photons))
        {
            // Sample wavelength and convert to energy
            real_type wave_length = sample_lambda(rng);
            CELER_EXPECT(wave_length > 0);
            photons_[i].energy = native_value_to<Energy>(
                constants::h_planck * constants::c_light / wave_length);

            // Sample direction
            real_type cost = sample_cost_(rng);
            real_type phi = sample_phi_(rng);
            photons_[i].direction = from_spherical(cost, phi);

            // Sample polarization perpendicular to the photon direction
            Real3 temp = from_spherical(
                (cost > 0 ? -1 : 1) * std::sqrt(1 - ipow<2>(cost)), phi);
            Real3 perp = {-std::sin(phi), std::cos(phi), 0};
            real_type sinphi, cosphi;
            sincospi(UniformRealDist{0, 1}(rng), &sinphi, &cosphi);
            for (int j = 0; j < 3; ++j)
            {
                photons_[i].polarization[j] = cosphi * temp[j]
                                              + sinphi * perp[j];
            }
            photons_[i].polarization
                = make_unit_vector(photons_[i].polarization);

            // Sample position
            real_type u = (is_neutral_) ? 1 : generate_canonical(rng);
            photons_[i].position = dist_.points[StepPoint::pre].pos
                                   + u * delta_pos_;

            // Sample time
            real_type delta_time
                = u * dist_.step_length
                  / (native_value_from(dist_.points[StepPoint::pre].speed)
                     + u * real_type(0.5) * native_value_from(delta_speed_));

            if (component.rise_time == 0)
            {
                delta_time -= component.fall_time
                              * std::log(generate_canonical(rng));
            }
            else
            {
                real_type scint_time{};
                real_type envelop{};
                do
                {
                    scint_time = sample_time(rng);
                    envelop = -std::expm1(-scint_time / component.rise_time);
                } while (!BernoulliDistribution(envelop)(rng));
                delta_time += scint_time;
            }
            CELER_ASSERT(delta_time >= 0);
            photons_[i].time = dist_.time + delta_time;
        }
        num_generated += num_photons;
    }
    CELER_ASSERT(num_generated == dist_.num_photons);

    return photons_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
