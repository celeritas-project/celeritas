//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/ScintillationGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "corecel/random/distribution/ExponentialDistribution.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "corecel/random/distribution/NormalDistribution.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "corecel/random/distribution/Selector.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/optical/detail/OpticalUtils.hh"

#include "GeneratorData.hh"
#include "ScintillationData.hh"
#include "../MaterialView.hh"
#include "../TrackInitializer.hh"

namespace celeritas
{
namespace optical
{
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
    ScintillationGenerator(MaterialView const&,
                           NativeCRef<ScintillationData> const& shared,
                           GeneratorDistributionData const& dist);

    // Construct without material (for testing)
    inline CELER_FUNCTION
    ScintillationGenerator(NativeCRef<ScintillationData> const& shared,
                           GeneratorDistributionData const& dist);

    // Sample a single photon from the distribution
    template<class Generator>
    inline CELER_FUNCTION TrackInitializer operator()(Generator& rng);

  private:
    //// TYPES ////

    using UniformRealDist = UniformRealDistribution<real_type>;
    using ExponentialDist = ExponentialDistribution<real_type>;

    //// DATA ////

    GeneratorDistributionData const& dist_;
    NativeCRef<ScintillationData> const& shared_;

    UniformRealDist sample_cost_;
    UniformRealDist sample_phi_;
    NormalDistribution<real_type> sample_lambda_;

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
    NativeCRef<ScintillationData> const& shared,
    GeneratorDistributionData const& dist)
    : dist_(dist)
    , shared_(shared)
    , sample_cost_(-1, 1)
    , sample_phi_(0, real_type(2 * constants::pi))
{
    if (shared_.scintillation_by_particle())
    {
        // TODO: implement sampling for particles
        CELER_ASSERT_UNREACHABLE();
    }

    CELER_EXPECT(dist_);
    CELER_EXPECT(shared_);

    auto const& pre_step = dist_.points[StepPoint::pre];
    auto const& post_step = dist_.points[StepPoint::post];
    delta_pos_ = post_step.pos - pre_step.pos;
    delta_speed_ = post_step.speed - pre_step.speed;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from shared scintillation data and distribution parameters.
 *
 * The optical material is unused but required for the Cherenkov and
 * scintillation generators to have the same signature.
 */
CELER_FUNCTION
ScintillationGenerator::ScintillationGenerator(
    MaterialView const&,
    NativeCRef<ScintillationData> const& shared,
    GeneratorDistributionData const& dist)
    : ScintillationGenerator(shared, dist)
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample a single scintillation photon.
 */
template<class Generator>
CELER_FUNCTION TrackInitializer ScintillationGenerator::operator()(Generator& rng)
{
    // Sample a component
    ScintRecord const& component = [&] {
        auto const& mat = shared_.materials[dist_.material];

        auto pdf = shared_.reals[mat.yield_pdf];
        auto select_idx = make_selector([&pdf](size_type i) { return pdf[i]; },
                                        mat.yield_pdf.size());
        size_type component_idx = select_idx(rng);
        CELER_ASSERT(component_idx < mat.components.size());
        return shared_.scint_records[mat.components[component_idx]];
    }();

    real_type energy_val{};
    if (!component.energy_cdf)
    {
        // Sample a photon for a single scintillation component, reusing the
        // "spare" value that the wavelength sampler might have stored
        CELER_ASSERT(component.lambda_mean > 0);
        sample_lambda_ = NormalDistribution{component.lambda_mean,
                                            component.lambda_sigma};
        real_type wavelength;
        do
        {
            // Rejecting the case of very large sigma and/or very small lambda
            wavelength = sample_lambda_(rng);
        } while (CELER_UNLIKELY(wavelength <= 0));
        energy_val = value_as<units::MevEnergy>(
            detail::wavelength_to_energy(wavelength));
    }
    else
    {
        // If the scintillation spectrum is provided use grid-based sampling
        NonuniformGridCalculator calc_cdf(
            shared_.energy_cdfs[component.energy_cdf], shared_.reals);

        NonuniformGridCalculator calc_energy = calc_cdf.make_inverse();
        energy_val = calc_energy(generate_canonical(rng));
    }

    ExponentialDist sample_time(real_type{1} / component.fall_time);

    TrackInitializer photon;
    photon.energy = Energy{energy_val};

    // Sample direction
    real_type cost = sample_cost_(rng);
    real_type phi = sample_phi_(rng);
    photon.direction = from_spherical(cost, phi);

    // Sample polarization perpendicular to the photon direction
    photon.polarization = [&] {
        Real3 pol = from_spherical(
            (cost > 0 ? -1 : 1) * std::sqrt(1 - ipow<2>(cost)), phi);
        Real3 perp = {-std::sin(phi), std::cos(phi), 0};
        real_type sinphi, cosphi;
        sincospi(UniformRealDist{}(rng), &sinphi, &cosphi);
        for (auto j : range(3))
        {
            pol[j] = cosphi * pol[j] + sinphi * perp[j];
        }
        // Enforce orthogonality
        return make_unit_vector(make_orthogonal(pol, photon.direction));
    }();
    CELER_ASSERT(is_soft_orthogonal(photon.polarization, photon.direction));

    // Sample the position
    real_type u = [&rng, &p = dist_.continuous_edep_fraction] {
        // The number of photons generated along the step (continuous energy
        // loss) and at the interaction site (local energy deposition) is
        // proportional to their respective energy contributions. If both
        // components are present, sample where to generate using the fraction
        // of the energy deposited along the step. The following condition is
        // statistically equivalent to sampling \c
        // BernoulliDistribution{p}(rng), but it avoids generating a random
        // number in the expected case where the probability is exactly zero or
        // one, while remaining correct if energy is deposited both along the
        // step and at the endpoint.
        if (p == 1 || (p != 0 && BernoulliDistribution{p}(rng)))
        {
            // Sample uniformly along the step
            return UniformRealDist{}(rng);
        }
        // Generate the photon at the discrete interaction site
        return real_type(1);
    }();
    photon.position = dist_.points[StepPoint::pre].pos;
    axpy(u, delta_pos_, &photon.position);

    // Sample time
    photon.time
        = dist_.time
          + u * dist_.step_length
                / (native_value_from(dist_.points[StepPoint::pre].speed)
                   + u * real_type(0.5) * native_value_from(delta_speed_));

    if (component.rise_time == 0)
    {
        // Sample exponentially from fall time
        photon.time += sample_time(rng);
    }
    else
    {
        real_type scint_time{};
        real_type target;
        do
        {
            // Sample time exponentially by fall time, then
            // accept with 1 - e^{-t/rise}
            scint_time = sample_time(rng);
            target = -std::expm1(-scint_time / component.rise_time);
        } while (RejectionSampler(target)(rng));
        photon.time += scint_time;
    }
    return photon;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
