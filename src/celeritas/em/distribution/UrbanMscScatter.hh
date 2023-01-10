// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/UrbanMscScatter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/grid/PolyEvaluator.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "UrbanMscHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample cos(theta) of the Urban multiple scattering model.
 */
class UrbanMscScatter
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using MscParameters = UrbanMscParameters;
    using MaterialData = UrbanMscMaterialData;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION UrbanMscScatter(UrbanMscRef const& shared,
                                          ParticleTrackView const& particle,
                                          GeoTrackView* geometry,
                                          PhysicsTrackView const& physics,
                                          MaterialView const& material,
                                          MscStep const& input,
                                          bool const geo_limited);

    // Sample the final true step length, position and direction by msc
    template<class Engine>
    inline CELER_FUNCTION MscInteraction operator()(Engine& rng);

  private:
    //// DATA ////

    real_type inc_energy_;
    Real3 inc_direction_;
    bool is_positron_;
    real_type rad_length_;
    real_type range_;
    real_type mass_;

    // Urban MSC parameters
    MscParameters const& params_;
    // Urban MSC material data
    MaterialData const& msc_;
    // Urban MSC helper class
    UrbanMscHelper helper_;
    // Results from UrbanMSCStepLimit
    bool is_displaced_;
    real_type geom_path_;
    real_type limit_min_;
    // Geomtry track view
    GeoTrackView& geometry_;

    real_type end_energy_;
    real_type lambda_;
    real_type true_path_;
    bool skip_sampling_;

    // Internal state
    real_type tau_{0};

    //// COMMON PROPERTIES ////

    //! The minimum step length for geometry 0.05 nm
    static CELER_CONSTEXPR_FUNCTION real_type geom_min()
    {
        return 5e-9 * units::centimeter;
    }

    //! The constant in the Highland theta0 formula
    static CELER_CONSTEXPR_FUNCTION Energy c_highland()
    {
        return units::MevEnergy{13.6};
    }

    //// HELPER FUNCTIONS ////

    // Calculate the true path length from the geom path length
    inline CELER_FUNCTION real_type calc_true_path(real_type true_path,
                                                   real_type geom_path,
                                                   real_type alpha) const;

    // Sample the angle, cos(theta), of the multiple scattering
    template<class Engine>
    inline CELER_FUNCTION real_type sample_cos_theta(Engine& rng,
                                                     real_type true_path,
                                                     real_type limit_min);

    // Sample consine(theta) with a large angle scattering
    template<class Engine>
    inline CELER_FUNCTION real_type simple_scattering(Engine& rng,
                                                      real_type xmean,
                                                      real_type x2mean) const;

    // Calculate the theta0 of the Highland formula
    inline CELER_FUNCTION real_type compute_theta0(real_type true_path) const;

    // Calculate the correction on theta0 for positrons
    inline CELER_FUNCTION real_type calc_correction(real_type tau) const;

    // Calculate the length of the displacement (using geometry safety)
    inline CELER_FUNCTION real_type calc_displacement_length(real_type rmax2);

    // Update direction and position after the multiple scattering
    template<class Engine>
    inline CELER_FUNCTION Real3 sample_displacement_dir(Engine& rng,
                                                        real_type phi) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
UrbanMscScatter::UrbanMscScatter(UrbanMscRef const& shared,
                                 ParticleTrackView const& particle,
                                 GeoTrackView* geometry,
                                 PhysicsTrackView const& physics,
                                 MaterialView const& material,
                                 MscStep const& input,
                                 bool const geo_limited)
    : inc_energy_(value_as<Energy>(particle.energy()))
    , inc_direction_(geometry->dir())
    , is_positron_(particle.particle_id() == shared.ids.positron)
    , rad_length_(material.radiation_length())
    , range_(physics.dedx_range())
    , mass_(value_as<Mass>(shared.electron_mass))
    , params_(shared.params)
    , msc_(shared.msc_data[material.material_id()])
    , helper_(shared, particle, physics)
    , is_displaced_(input.is_displaced && !geo_limited)
    , geom_path_(input.geom_path)
    , limit_min_(physics.msc_range().limit_min)
    , geometry_(*geometry)
{
    CELER_EXPECT(particle.particle_id() == shared.ids.electron
                 || particle.particle_id() == shared.ids.positron);
    CELER_EXPECT(geom_path_ > 0);

    lambda_ = helper_.msc_mfp(Energy{inc_energy_});

    // Convert the geometry path length to the true path length if needed
    true_path_ = !geo_limited ? input.true_path
                              : this->calc_true_path(
                                  input.true_path, geom_path_, input.alpha);

    // Protect against a wrong true -> geom -> true transformation
    true_path_ = min<real_type>(true_path_, input.phys_step);
    CELER_ASSERT(true_path_ >= geom_path_);

    skip_sampling_ = true;
    if (true_path_ < range_ && true_path_ > params_.geom_limit)
    {
        end_energy_ = value_as<Energy>(helper_.calc_end_energy(true_path_));
        skip_sampling_
            = (end_energy_ < value_as<Energy>(params_.min_sampling_energy())
               || true_path_ <= shared.params.limit_min_fix()
               || true_path_ < lambda_ * params_.tau_small);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Sample the angular distribution and the lateral displacement by multiple
 * scattering as well as convert the geometrical path length to the true path
 * length based on G4VMultipleScattering::AlongStepDoIt and
 * G4UrbanMscModel::SampleScattering of the Geant4 10.7 release.
 */
template<class Engine>
CELER_FUNCTION auto UrbanMscScatter::operator()(Engine& rng) -> MscInteraction
{
    if (skip_sampling_)
    {
        // Do not sample scattering at the last or at a small step
        return {true_path_,
                inc_direction_,
                {0, 0, 0},
                MscInteraction::Action::unchanged};
    }

    // Sample polar angle and update tau_
    real_type costheta = this->sample_cos_theta(rng, true_path_, limit_min_);
    CELER_ASSERT(std::fabs(costheta) <= 1);

    // Sample azimuthal angle, used for displacement and exiting angle
    real_type phi
        = UniformRealDistribution<real_type>(0, 2 * constants::pi)(rng);

    MscInteraction result;
    result.action = MscInteraction::Action::scattered;
    {
        // This should only be needed to silence compiler warning
        result.displacement = {0, 0, 0};
    }

    // Calculate displacement
    if (is_displaced_ && tau_ >= params_.tau_small)
    {
        // Sample displacement and adjust
        real_type length = this->calc_displacement_length(
            (true_path_ - geom_path_) * (true_path_ + geom_path_));
        if (length > 0)
        {
            result.displacement = this->sample_displacement_dir(rng, phi);
            for (int i = 0; i < 3; ++i)
            {
                result.displacement[i] *= length;
            }
            result.action = MscInteraction::Action::displaced;
        }
    }

    // Calculate direction and return
    result.step_length = true_path_;
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
CELER_FUNCTION real_type UrbanMscScatter::sample_cos_theta(Engine& rng,
                                                           real_type true_path,
                                                           real_type limit_min)
{
    using PolyQuad = PolyEvaluator<real_type, 2>;

    real_type result = 1;

    real_type lambda_end = helper_.msc_mfp(Energy{end_energy_});

    tau_ = true_path
           / ((std::fabs(lambda_ - lambda_end) > lambda_ * real_type(0.01))
                  ? (lambda_ - lambda_end) / std::log(lambda_ / lambda_end)
                  : lambda_);

    if (tau_ >= params_.tau_big)
    {
        result = UniformRealDistribution<real_type>(-1, 1)(rng);
    }
    else if (tau_ >= params_.tau_small)
    {
        // Sample the mean distribution of the scattering angle, cos(theta)

        // Eq. 8.2 and \f$ \cos^2\theta \f$ term in Eq. 8.3 in PRM
        real_type xmean = std::exp(-tau_);
        real_type x2mean = (1 + 2 * std::exp(real_type(-2.5) * tau_)) / 3;

        // Too large step of the low energy particle
        if (end_energy_ < real_type(0.5) * inc_energy_)
        {
            return this->simple_scattering(rng, xmean, x2mean);
        }

        // Check for extreme small steps
        real_type tsmall = min<real_type>(limit_min, params_.lambda_limit);
        bool small_step = (true_path < tsmall);

        real_type theta0 = (small_step) ? std::sqrt(true_path / tsmall)
                                              * this->compute_theta0(tsmall)
                                        : this->compute_theta0(true_path);

        // Protect for very small angles
        real_type theta2 = ipow<2>(theta0);
        if (theta2 < params_.tau_small)
        {
            return result;
        }

        if (theta0 > constants::pi / 6)
        {
            // theta0 > theta0_max
            return this->simple_scattering(rng, xmean, x2mean);
        }

        real_type x = theta2 * (1 - theta2 / 12);
        if (theta2 > real_type(0.01))
        {
            x = ipow<2>(2 * std::sin(real_type(0.5) * theta0));
        }

        // Evaluate parameters for the tail distribution
        real_type u
            = fastpow(small_step ? tsmall / lambda_ : tau_, 1 / real_type(6));
        real_type xsi = PolyQuad(msc_.d[0], msc_.d[1], msc_.d[2])(u)
                        + msc_.d[3]
                              * std::log(true_path / (tau_ * rad_length_));

        // The tail should not be too big
        xsi = max<real_type>(xsi, real_type(1.9));

        real_type c = xsi;
        if (std::fabs(xsi - 3) < real_type(0.001))
        {
            c = real_type(3.001);
        }
        else if (std::fabs(xsi - 2) < real_type(0.001))
        {
            c = real_type(2.001);
        }

        real_type ea = std::exp(-xsi);
        // Mean of cos\theta computed from the distribution g_1(cos\theta)
        real_type xmean1 = 1 - (1 - (1 + xsi) * ea) * x / (1 - ea);

        if (xmean1 <= real_type(0.999) * xmean)
        {
            result = this->simple_scattering(rng, xmean, x2mean);
        }

        // From continuity of derivatives
        real_type b1 = 2 + (c - xsi) * x;
        real_type d = fastpow(c * x / b1, c - 1);
        real_type x0 = 1 - xsi * x;

        // Mean of cos\theta computed from the distribution g_2(cos\theta)
        real_type xmean2 = (x0 + d - (c * x - b1 * d) / (c - 2)) / (1 - d);

        real_type f2x0 = (c - 1) / (c * (1 - d));
        real_type prob = f2x0 / (ea / (1 - ea) + f2x0);

        // Eq. 8.14 in the PRM: note that can be greater than 1
        real_type qprob = xmean / (prob * xmean1 + (1 - prob) * xmean2);

        // Sampling of cos(theta)
        if (generate_canonical(rng) < qprob)
        {
            // Note: prob is sometime a little greater than one
            if (generate_canonical(rng) < prob)
            {
                // Sample \f$ \cos\theta \f$ from \f$ g_1(\cos\theta) \f$
                UniformRealDistribution<real_type> sample_inner(ea, 1);
                result = 1 + std::log(sample_inner(rng)) * x;
            }
            else
            {
                // Sample \f$ \cos\theta \f$ from \f$ g_2(\cos\theta) \f$
                real_type var = (1 - d) * generate_canonical(rng);
                if (var < real_type(0.01) * d)
                {
                    var /= (d * (c - 1));
                    result = -1
                             + var * (1 - real_type(0.5) * var * c)
                                   * (2 + (c - xsi) * x);
                }
                else
                {
                    result = x * (c - xsi - c * fastpow(var + d, -1 / (c - 1)))
                             + 1;
                }
            }
        }
        else
        {
            // Sample \f$ \cos\theta \f$ from \f$ g_3(\cos\theta) \f$
            result = UniformRealDistribution<real_type>(-1, 1)(rng);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sample the large angle scattering using 2 model functions.
 *
 * \param rng Random number generator
 * \param xmean the mean of \f$\cos\theta\f$.
 * \param x2mean the mean of \f$\cos\theta^{2}\f$.
 */
template<class Engine>
CELER_FUNCTION real_type UrbanMscScatter::simple_scattering(
    Engine& rng, real_type xmean, real_type x2mean) const
{
    real_type a = (2 * xmean + 9 * x2mean - 3) / (2 * xmean - 3 * x2mean + 1);
    BernoulliDistribution sample_pow{(a + 2) * xmean / a};

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
 * using a modified Highland-Lynch-Dahl formula. All particles take the width
 * of the central part from a parameterization similar to the original Highland
 * formula, Particle Physics Booklet, July 2002, eq. 26.10.
 * \f[
 *   \theta_0 = \frac{13.6\rm{MeV}}{\beta c p} z_{ch} \sqrt{\ln(t/X_o)} c
 * \f]
 * where \f$p, \beta\c, z_{ch}\f$, \f$t/X_0\f$ and \f$c\f$ are the momentum,
 * velocity, charge number of the incident particle, the true path length in
 * radiation length unit and the correction term, respectively. For details,
 * see the section 8.1.5 of the Geant4 10.7 Physics Reference Manual.
 *
 * \param true_path the true step length.
 */
CELER_FUNCTION
real_type UrbanMscScatter::compute_theta0(real_type true_path) const
{
    real_type invbetacp
        = std::sqrt((inc_energy_ + mass_) * (end_energy_ + mass_)
                    / (inc_energy_ * (inc_energy_ + 2 * mass_) * end_energy_
                       * (end_energy_ + 2 * mass_)));
    real_type y = true_path / rad_length_;

    // Correction for the positron
    if (is_positron_)
    {
        real_type tau = std::sqrt(inc_energy_ * end_energy_) / mass_;
        y *= this->calc_correction(tau);
    }

    // Note: multiply abs(charge) if the charge number is not unity
    real_type theta0 = value_as<Energy>(c_highland()) * std::sqrt(y)
                       * invbetacp;

    // Correction factor from e- scattering data
    theta0 *= (msc_.coeffth1 + msc_.coeffth2 * std::log(y));

    return theta0;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the correction on theta0 for positrons.
 *
 * \param tau (incident energy * energy at the end of step)/electron_mass.
 */
CELER_FUNCTION real_type UrbanMscScatter::calc_correction(real_type tau) const
{
    using PolyLin = PolyEvaluator<real_type, 1>;
    using PolyQuad = PolyEvaluator<real_type, 2>;

    real_type corr{1.0};

    real_type zeff = msc_.zeff;
    constexpr real_type xl = 0.6;
    constexpr real_type xh = 0.9;
    constexpr real_type e = 113;

    real_type x = std::sqrt(tau * (tau + 2) / ipow<2>(tau + 1));
    real_type a = PolyLin(0.994, -4.08e-3)(zeff);
    real_type b = PolyQuad(7.16, 52.6, 365)(1 / zeff);
    real_type c = PolyLin(1, -4.47e-3)(zeff);
    real_type d = real_type(1.21e-3) * zeff;
    if (x < xl)
    {
        corr = a * (1 - std::exp(-b * x));
    }
    else if (x > xh)
    {
        corr = c + d * std::exp(e * (x - 1));
    }
    else
    {
        real_type yl = a * (1 - std::exp(-b * xl));
        real_type yh = c + d * std::exp(e * (xh - 1));
        real_type y0 = (yh - yl) / (xh - xl);
        real_type y1 = yl - y0 * xl;
        corr = y0 * x + y1;
    }

    corr *= PolyQuad(1.41125, -1.86427e-2, 1.84035e-4)(zeff);

    return corr;
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
/*!
 * Scale displacement and correct near the boundary.
 *
 * This is a transformation of the logic in geant4, with \c fPositionChanged
 * being equal to `rho == 0`. The key takeaway for the displacement calculation
 * is that for small displacement values, *or* for small safety distances, we
 * do not displace. For large safety distances we do not displace further than
 * the safety.
 */
CELER_FUNCTION real_type
UrbanMscScatter::calc_displacement_length(real_type rmax2)
{
    CELER_EXPECT(rmax2 >= 0);

    // 0.73 is (roughly) the expected value of a distribution of the mean
    // radius given rmax "based on single scattering results"
    // https://github.com/Geant4/geant4/blame/28a70706e0edf519b16e864ebf1d2f02a00ba596/source/processes/electromagnetic/standard/src/G4UrbanMscModel.cc#L1142
    constexpr real_type mean_radius_frac{0.73};

    real_type rho = mean_radius_frac * std::sqrt(rmax2);

    if (rho <= params_.geom_limit)
    {
        // Displacement is too small to bother with
        rho = 0;
    }
    else
    {
        real_type safety = (1 - params_.safety_tol) * geometry_.find_safety();
        if (safety <= params_.geom_limit)
        {
            // We're near a volume boundary so do not displace at all
            rho = 0;
        }
        else
        {
            // Do not displace further than safety
            rho = min(rho, safety);
        }
    }

    return rho;
}

//---------------------------------------------------------------------------//
/*!
 * Compute the true path length for a given geom path (the z -> t conversion).
 *
 * The transformation can be written as
 * \f[
 *     t(z) = \langle t \rangle = -\lambda_{1} \log(1 - \frac{z}{\lambda_{1}})
 * \f]
 * or \f$ t(z) = \frac{1}{\alpha} [ 1 - (1-\alpha w z)^{1/w}] \f$ if the
 * geom path is small, where \f$ w = 1 + \frac{1}{\alpha \lambda_{10}}\f$.
 *
 * \param true_path the proposed step before transportation.
 * \param geom_path the proposed step after transportation.
 * \param alpha variable from UrbanMscStepLimit.
 */
CELER_FUNCTION
real_type UrbanMscScatter::calc_true_path(real_type true_path,
                                          real_type geom_path,
                                          real_type alpha) const
{
    CELER_EXPECT(geom_path <= true_path);
    if (geom_path < params_.min_step())
    {
        // geometrical path length = true path length for a very small step
        return geom_path;
    }

    // Recalculation
    real_type length = geom_path;

    // NOTE: add && !insideskin if the UseDistanceToBoundary algorithm is used
    if (geom_path > lambda_ * params_.tau_small)
    {
        if (alpha < 0)
        {
            // For cases that the true path is very small compared to either
            // the mean free path or the range
            length = -lambda_ * std::log(1 - geom_path / lambda_);
        }
        else
        {
            real_type w = 1 + 1 / (alpha * lambda_);
            real_type x = alpha * w * geom_path;
            length = (x < 1) ? (1 - fastpow(1 - x, 1 / w)) / alpha : range_;
        }

        length = clamp(length, geom_path, true_path);
    }

    return length;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
