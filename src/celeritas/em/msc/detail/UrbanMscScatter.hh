// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/msc/detail/UrbanMscScatter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/PolyEvaluator.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "UrbanMscHelper.hh"
#include "UrbanPositronCorrector.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample angular change and lateral displacement with the Urban multiple
 * scattering model.

 * \note This code performs the same method as in
 * \c G4VMultipleScattering::AlongStepDoIt
 * and \c G4UrbanMscModel::SampleScattering of the Geant4 10.7 release.
 */
class UrbanMscScatter
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using UrbanMscRef = NativeCRef<UrbanMscData>;
    //!@}

  public:
    // TODO: improve this
    static inline CELER_FUNCTION real_type
    calc_displacement(real_type true_path, real_type geom_path);

    // Construct with shared and state data
    inline CELER_FUNCTION UrbanMscScatter(UrbanMscRef const& shared,
                                          UrbanMscHelper const& helper,
                                          ParticleTrackView const& particle,
                                          PhysicsTrackView const& physics,
                                          MaterialView const& material,
                                          Real3 const& dir,
                                          real_type safety,
                                          MscStep const& input);

    // Sample the final true step length, position and direction by msc
    template<class Engine>
    inline CELER_FUNCTION MscInteraction operator()(Engine& rng);

  private:
    //// DATA ////

    // Shared constant data
    UrbanMscRef const& shared_;
    // Urban MSC material data
    UrbanMscMaterialData const& msc_;
    // Urban MSC helper class
    UrbanMscHelper const& helper_;
    // Material data
    MaterialView const& material_;

    real_type inc_energy_;
    Real3 const& inc_direction_;
    real_type safety_;

    // Results from UrbanMSCStepLimit
    bool is_displaced_;
    real_type geom_path_;
    real_type true_path_;
    real_type limit_min_;

    // Calculated values for sampling
    bool skip_sampling_;
    real_type end_energy_;
    real_type tau_{0};
    real_type xmean_{0};
    real_type x2mean_{0};
    real_type theta0_{-1};

    //// COMMON PROPERTIES ////

    //! The minimum step length for geometry 0.05 nm
    static CELER_CONSTEXPR_FUNCTION real_type geom_min()
    {
        return 5e-9 * units::centimeter;
    }

    //// HELPER FUNCTIONS ////

    // Sample the angle, cos(theta), of the multiple scattering
    template<class Engine>
    inline CELER_FUNCTION real_type sample_cos_theta(Engine& rng) const;

    // Sample consine(theta) with a large angle scattering
    template<class Engine>
    inline CELER_FUNCTION real_type simple_scattering(Engine& rng) const;

    // Calculate the theta0 of the Highland formula
    inline CELER_FUNCTION real_type
    compute_theta0(ParticleTrackView const& particle) const;

    // Update direction and position after the multiple scattering
    template<class Engine>
    inline CELER_FUNCTION Real3 sample_displacement_dir(Engine& rng,
                                                        real_type phi) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
CELER_FUNCTION real_type UrbanMscScatter::calc_displacement(real_type geom_path,
                                                            real_type true_path)
{
    CELER_EXPECT(true_path >= geom_path);

    // true^2 - geo^2
    real_type rmax2 = diffsq(true_path, geom_path);

    // 0.73 is (roughly) the expected value of a distribution of the mean
    // radius given rmax "based on single scattering results"
    // https://github.com/Geant4/geant4/blame/28a70706e0edf519b16e864ebf1d2f02a00ba596/source/processes/electromagnetic/standard/src/G4UrbanMscModel.cc#L1142
    constexpr real_type mean_radius_frac{0.73};

    return mean_radius_frac * std::sqrt(rmax2);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * This function also precalculates distribution-independent quantities, e.g.
 * converting the geometrical path length to the true path.
 */
CELER_FUNCTION
UrbanMscScatter::UrbanMscScatter(UrbanMscRef const& shared,
                                 UrbanMscHelper const& helper,
                                 ParticleTrackView const& particle,
                                 PhysicsTrackView const& physics,
                                 MaterialView const& material,
                                 Real3 const& dir,
                                 real_type safety,
                                 MscStep const& input)
    : shared_(shared)
    , msc_(shared.material_data[material.material_id()])
    , helper_(helper)
    , material_(material)
    , inc_energy_(value_as<Energy>(particle.energy()))
    , inc_direction_(dir)
    , safety_(safety)
    , is_displaced_(input.is_displaced)
    , geom_path_(input.geom_path)
    , true_path_(input.true_path)
    , limit_min_(physics.msc_range().limit_min)
{
    CELER_EXPECT(safety_ >= 0);
    CELER_EXPECT(geom_path_ > 0);
    CELER_EXPECT(true_path_ >= geom_path_);
    CELER_EXPECT(limit_min_ >= UrbanMscParameters::limit_min_fix()
                 || !is_displaced_);
    CELER_EXPECT(!is_displaced_ || safety > 0);

    skip_sampling_ = [this, &helper, &physics] {
        if (true_path_ == physics.dedx_range())
        {
            // Range-limited step (particle stops)
            // TODO: probably redundant with low 'end energy'
            return true;
        }
        if (true_path_ < shared_.params.geom_limit)
        {
            // Very small step (NOTE: with the default values in UrbanMscData,
            // this is redundant witih the tau_small comparison below if MFP >=
            // 0.005 cm)
            return true;
        }

        // Lazy calculation of end energy
        end_energy_ = value_as<Energy>(helper.calc_end_energy(true_path_));

        if (Energy{end_energy_} < shared_.params.min_sampling_energy())
        {
            // Ending energy is very low
            return true;
        }
        if (true_path_ <= helper_.msc_mfp() * shared_.params.tau_small)
        {
            // Very small MFP travelled
            return true;
        }
        return false;
    }();

    // TODO: there are several different sampling strategies for angle change:
    // - very small step/very low energy endpoint: no scattering
    // - very small mfp: (probably impossible because of condition above):
    //   forward scatter
    // - very large mfp: exiting angle is isotropic
    // - large energy loss: "simple_scattering"

    if (!skip_sampling_)
    {
        // Calculate number of mean free paths traveled
        tau_ = true_path_ / [this, &helper] {
            // Calculate the average MFP assuming the cross section varies
            // linearly over the step
            real_type lambda = helper_.msc_mfp();
            real_type lambda_end = helper.calc_msc_mfp(Energy{end_energy_});
            if (std::fabs(lambda - lambda_end) < lambda * real_type(0.01))
            {
                // Cross section is almost constant over the step: avoid
                // numerical explosion
                return helper_.msc_mfp();
            }
            return (lambda - lambda_end) / std::log(lambda / lambda_end);
        }();

        if (tau_ < shared_.params.tau_big)
        {
            // Eq. 8.2 and \f$ \cos^2\theta \f$ term in Eq. 8.3 in PRM
            xmean_ = std::exp(-tau_);
            x2mean_ = (1 + 2 * std::exp(real_type(-2.5) * tau_)) / 3;

            // MSC "true path" step limit
            if (CELER_UNLIKELY(limit_min_ == 0))
            {
                // Unlikely: MSC range cache wasn't initialized by
                // UrbanMscStepLimit
                CELER_ASSERT(!is_displaced_);
                limit_min_ = UrbanMscParameters::limit_min();
            }
            limit_min_ = min(limit_min_, physics.scalars().lambda_limit);

            // TODO: theta0_ calculation could be done externally, eliminating
            // many of the class member data
            theta0_ = this->compute_theta0(particle);

            if (theta0_ < real_type(1e-8))
            {
                // Arbitrarily (?) small angle change (theta_0^2 < 1e-16):
                // skip sampling angular distribution if width of direction
                // distribution is too narrow
                if (!is_displaced_)
                {
                    // No angular sampling and no displacement => no change
                    skip_sampling_ = true;
                }
                else
                {
                    theta0_ = 0;
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Sample the angular distribution and the lateral displacement by multiple
 * scattering.
 */
template<class Engine>
CELER_FUNCTION auto UrbanMscScatter::operator()(Engine& rng) -> MscInteraction
{
    if (skip_sampling_)
    {
        // Do not sample scattering at the last or at a small step
        return {inc_direction_, {0, 0, 0}, MscInteraction::Action::unchanged};
    }

    // Sample polar angle cosine
    real_type costheta = [this, &rng] {
        if (theta0_ <= 0)
        {
            // Very small outgoing angular distribution
            return real_type{1};
        }
        if (tau_ >= shared_.params.tau_big)
        {
            // Long mean free path: exiting direction is isotropic
            UniformRealDistribution<real_type> sample_isotropic(-1, 1);
            return sample_isotropic(rng);
        }
        if (2 * end_energy_ < inc_energy_ || theta0_ > constants::pi / 6)
        {
            // Large energy loss over the step or large angle distribution
            // width
            return this->simple_scattering(rng);
        }
        // No special cases match:
        return this->sample_cos_theta(rng);
    }();
    CELER_ASSERT(std::fabs(costheta) <= 1);

    // Sample azimuthal angle, used for displacement and exiting angle
    real_type phi = UniformRealDistribution<real_type>(
        0, real_type{2 * constants::pi})(rng);

    MscInteraction result;
    result.action = MscInteraction::Action::scattered;
    // This should only be needed to silence compiler warning, since the
    // displacement should be ignored since our action result is
    // 'scattered'
    result.displacement = {0, 0, 0};

    if (is_displaced_)
    {
        // Calculate displacement length
        real_type length = this->calc_displacement(geom_path_, true_path_);
        // Don't displace further than safety (minus a tolerance)
        length = min(length, (1 - shared_.params.safety_tol) * safety_);

        if (length >= shared_.params.geom_limit)
        {
            // Displacement distance is large enough to worry about
            result.displacement = this->sample_displacement_dir(rng, phi);
            result.displacement *= length;
            result.action = MscInteraction::Action::displaced;
        }
    }

    // Calculate direction and return
    result.direction = rotate(from_spherical(costheta, phi), inc_direction_);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sample the scattering angle at the end of the true step length.
 *
 * The scattering angle \f$\theta\f$ and true step length, \f$t\f$ are
 * described in G4UrbanMscModel::SampleCosineTheta of the
 * Geant4 10.7 release. See also, CERN-OPEN-2006-077 by L. Urban.
 *
 * The mean value of \f$u = \cos\theta\f$ follows \f$\exp(-t/\lambda_{1})\f$
 * and the variance is written as \f$\frac{1+2e^{-\kappa r}}{3} - e^{-2r}\f$
 * where \f$r = t/\lambda_{1}\f$ and \f$\kappa = \lambda_{1}/\lambda_{2}\f$.
 * The \f$\cos\theta\f$ is sampled according to a model function of \f$u\f$,
 * \f[
 *   g(u) = q [ p g_1(u) + (1-p) g_2(u) ] - (1 - q) g_3(u)
 * \f]
 * where \f$p,q \in [0,1]\f$ and the functions \f$g_i\f$ have been chosen as
 * \f[
 *   g_1(u) = c_1 e^{-a(1-u)},
 *   g_2(u) = \frac{c_2}{(b-u)^d},
 *   g_3(u) = c_3
 * \f]
 * with normalization constants, \f$d\f$. For small angles, \f$g_1\f$ is
 * nearly Gaussian, \f$ \exp(-\frac{\theta^{2}}{2\theta_{0}^{2}}), \f$
 * if \f$\theta_0 \approx 1/1\f$, while \f$g_2\f$ has a Rutherford-like tail
 * for large \f$\theta\f$, if \f$b \approx 1\f$ and \f$d\f$ is not far from 2.
 */
template<class Engine>
CELER_FUNCTION real_type UrbanMscScatter::sample_cos_theta(Engine& rng) const
{
    CELER_EXPECT(xmean_ > 0 && x2mean_ > 0);

    // Evaluate parameters for the tail distribution
    real_type xsi = [this] {
        using PolyQuad = PolyEvaluator<real_type, 2>;

        real_type maxtau
            = true_path_ < limit_min_ ? limit_min_ / helper_.msc_mfp() : tau_;
        // note: 0 < u <= sqrt(2) when shared_.params.tau_big == 8
        real_type u = fastpow(maxtau, 1 / real_type(6));
        // Number of radiation lengths traveled by the average MFP over this
        // step
        real_type radlen_mfp = true_path_
                               / (tau_ * material_.radiation_length());
        real_type result = PolyQuad(msc_.tail_coeff)(u)
                           + msc_.tail_corr * std::log(radlen_mfp);
        // The tail should not be too big
        return max(result, real_type(1.9));
    }();

    real_type ea = std::exp(-xsi);
    // Mean of cos\theta computed from the distribution g_1(cos\theta)
    // small theta => x = theta0^2
    // large xsi => xmean_1 = 1 - x
    // small tau => xmean = 1
    real_type x = ipow<2>(2 * std::sin(real_type(0.5) * theta0_));
    real_type xmean_1 = 1 - x * (1 + (xsi * ea) / (1 - ea));

    if (xmean_1 <= real_type(0.999) * xmean_)
    {
        return this->simple_scattering(rng);
    }

    // From continuity of derivatives
    real_type c = [xsi] {
        if (std::fabs(xsi - 3) < real_type(0.001))
        {
            return real_type(3.001);
        }
        else if (std::fabs(xsi - 2) < real_type(0.001))
        {
            return real_type(2.001);
        }
        return xsi;
    }();
    real_type b1 = 2 + (c - xsi) * x;
    real_type d = fastpow(c * x / b1, c - 1);
    real_type x0 = 1 - xsi * x;

    // Mean of cos\theta computed from the distribution g_2(cos\theta)
    real_type xmean_2 = (x0 + d - (c * x - b1 * d) / (c - 2)) / (1 - d);

    real_type f2x0 = (c - 1) / (c * (1 - d));
    real_type prob = f2x0 / (ea / (1 - ea) + f2x0);

    // Eq. 8.14 in the PRM: note that can be greater than 1
    real_type qprob = xmean_ / (prob * xmean_1 + (1 - prob) * xmean_2);
    // Sampling of cos(theta)
    if (generate_canonical(rng) >= qprob)
    {
        // Sample \f$ \cos\theta \f$ from \f$ g_3(\cos\theta) \f$
        return UniformRealDistribution<real_type>(-1, 1)(rng);
    }

    // Note: prob is sometime a little greater than one
    if (generate_canonical(rng) < prob)
    {
        // Sample \f$ \cos\theta \f$ from \f$ g_1(\cos\theta) \f$
        UniformRealDistribution<real_type> sample_inner(ea, 1);
        return 1 + std::log(sample_inner(rng)) * x;
    }
    else
    {
        // Sample \f$ \cos\theta \f$ from \f$ g_2(\cos\theta) \f$
        real_type var = (1 - d) * generate_canonical(rng);
        if (var < real_type(0.01) * d)
        {
            var /= (d * (c - 1));
            return -1
                   + var * (1 - real_type(0.5) * var * c) * (2 + (c - xsi) * x);
        }
        else
        {
            return x * (c - xsi - c * fastpow(var + d, -1 / (c - 1))) + 1;
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Sample the large angle scattering using 2 model functions.
 *
 * \param rng Random number generator
 */
template<class Engine>
CELER_FUNCTION real_type UrbanMscScatter::simple_scattering(Engine& rng) const
{
    real_type a = (2 * xmean_ + 9 * x2mean_ - 3)
                  / (2 * xmean_ - 3 * x2mean_ + 1);
    BernoulliDistribution sample_pow{(a + 2) * xmean_ / a};

    // Sample cos(theta)
    real_type result{};
    do
    {
        real_type rdm = generate_canonical(rng);
        result = 2 * (sample_pow(rng) ? fastpow(rdm, 1 / (a + 1)) : rdm) - 1;
    } while (std::fabs(result) > 1);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the width of an approximate Gaussian projected angle distribution
 * using a modified Highland-Lynch-Dahl formula.
 *
 * All particles take the width
 * of the central part from a parameterization similar to the original Highland
 * formula, Particle Physics Booklet, July 2002, eq. 26.10.
 * \f[
 *   \theta_0 = \frac{13.6\rm{MeV}}{\beta c p} z_{ch} \sqrt{\ln(t/X_o)} c
 * \f]
 * where \f$p, \beta\c, z_{ch}\f$, \f$t/X_0\f$ and \f$c\f$ are the momentum,
 * velocity, charge number of the incident particle, the true path length in
 * radiation length unit and the correction term, respectively. For details,
 * see the section 8.1.5 of the Geant4 10.7 Physics Reference Manual.
 */
CELER_FUNCTION
real_type
UrbanMscScatter::compute_theta0(ParticleTrackView const& particle) const
{
    real_type const mass = value_as<Mass>(shared_.electron_mass);
    real_type true_path = max(limit_min_, true_path_);
    real_type y = true_path / material_.radiation_length();

    // Correction for the positron
    if (particle.particle_id() == shared_.ids.positron)
    {
        detail::UrbanPositronCorrector calc_correction{material_.zeff()};
        y *= calc_correction(std::sqrt(inc_energy_ * end_energy_) / mass);
    }
    CELER_ASSERT(y > 0);

    real_type invbetacp
        = std::sqrt((inc_energy_ + mass) * (end_energy_ + mass)
                    / (inc_energy_ * (inc_energy_ + 2 * mass) * end_energy_
                       * (end_energy_ + 2 * mass)));
    constexpr units::MevEnergy c_highland{13.6};
    real_type theta0
        = c_highland.value()
          * std::abs(value_as<units::ElementaryCharge>(particle.charge()))
          * std::sqrt(y) * invbetacp;

    // Correction factor from e- scattering data
    theta0 *= PolyEvaluator<real_type, 1>(msc_.theta_coeff)(std::log(y));

    if (true_path_ < limit_min_)
    {
        // Apply correction if true path is very small
        theta0 *= std::sqrt(true_path_ / limit_min_);
    }

    // Very small path lengths can result in a negative e- scattering
    // correction: clamp to zero so that too-small paths result in no change
    // in angle
    return max<real_type>(theta0, 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the displacement direction using G4UrbanMscModel::SampleDisplacement
 * (simple and fast sampling based on single scattering results) and update
 * direction and position of the particle.
 *
 * A simple distribution for the unit direction on the lateral (x-y) plane,
 * \f$ Phi = \phi \pm \psi \f$ where \f$ psi \sim \exp(-\beta*v) \f$ and
 * \f$\beta\f$ is determined from the requirement that the distribution should
 * give the same mean value that is obtained from the single scattering
 * simulation.
 *
 * \param rng Random number generator
 * \param phi the azimuthal angle of the multiple scattering.
 */
template<class Engine>
CELER_FUNCTION Real3
UrbanMscScatter::sample_displacement_dir(Engine& rng, real_type phi) const
{
    // Sample a unit direction of the displacement
    constexpr real_type cbeta = 2.160;
    // cbeta1 = 1 - std::exp(-cbeta * constants::pi);
    constexpr real_type cbeta1 = 0.9988703417569197;

    real_type psi = -std::log(1 - generate_canonical(rng) * cbeta1) / cbeta;
    phi += BernoulliDistribution(0.5)(rng) ? psi : -psi;

    Real3 displacement{std::cos(phi), std::sin(phi), 0};

    // Rotate along the incident particle direction
    displacement = rotate(displacement, inc_direction_);
    return displacement;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
