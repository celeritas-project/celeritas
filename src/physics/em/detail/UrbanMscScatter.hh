// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UrbanMscScatter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Algorithms.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/material/Types.hh"

#include "UrbanMscData.hh"
#include "UrbanMscHelper.hh"
#include "UrbanMscStepLimit.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Output data type of UrbanMscScatter.
 */
struct MscScatterResult
{
    real_type step_length;  //!< true step length
    Real3     direction;    //!< final direction by msc
    Real3     displacement; //!< lateral displacement
};

//---------------------------------------------------------------------------//
/*!
 * This is a class for sampling cos(theta) of the Urban multiple scattering.
 */
class UrbanMscScatter
{
  public:
    //!@{
    //! Type aliases
    using Energy          = units::MevEnergy;
    using StepLimitResult = detail::MscStepLimitResult;
    using MscParameters   = detail::UrbanMscParameters;
    using MaterialData    = detail::UrbanMscMaterialData;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION UrbanMscScatter(const UrbanMscNativeRef& shared,
                                          const ParticleTrackView& particle,
                                          GeoTrackView*            geometry,
                                          const PhysicsTrackView&  physics,
                                          const MaterialView&      material,
                                          const StepLimitResult&   input);

    // Sample the final true step length, position and direction by msc
    template<class Engine>
    inline CELER_FUNCTION MscScatterResult operator()(Engine& rng);

  private:
    //// DATA ////

    Energy    inc_energy_;
    Real3     inc_direction_;
    bool      is_positron_;
    real_type rad_length_;
    real_type mass_;
    real_type lambda_;
    real_type tau_{0};

    // Urban MSC parameters
    const MscParameters& params_;
    // Urban MSC material data
    const MaterialData& msc_;
    // Urban MSC helper class
    UrbanMscHelper helper_;
    // Results from UrbanMSCStepLimit
    StepLimitResult input_;
    // Geomtry track view
    GeoTrackView& geometry_;

    //// COMMON PROPERTIES ////

    //! The minimum step length for geometry ( 0.05*CLHEP::nm)
    static CELER_CONSTEXPR_FUNCTION real_type geom_min()
    {
        return 5e-9 * units::centimeter;
    }

    //! The constant in the Highland theta0 formula: 13.6 MeV
    static CELER_CONSTEXPR_FUNCTION Energy c_highland()
    {
        return Energy{13.6};
    }

    //// HELPER FUNCTIONS ////

    // Sample the angle, cos(theta), of the multiple scattering
    template<class Engine>
    inline CELER_FUNCTION real_type sample_cos_theta(Engine&   rng,
                                                     Energy    end_energy,
                                                     real_type true_path,
                                                     real_type limit_min);

    // Sample consine(theta) with a large angle scattering
    template<class Engine>
    inline CELER_FUNCTION real_type simple_scattering(Engine&   rng,
                                                      real_type xmean,
                                                      real_type x2mean) const;

    // Calculate the theta0 of the Highland formula
    inline CELER_FUNCTION real_type compute_theta0(real_type true_path,
                                                   Energy    end_energy) const;

    // Calculate the correction on theta0 for positrons
    inline CELER_FUNCTION real_type calc_correction(real_type tau) const;

    // Update direction and position after the multiple scattering
    template<class Engine>
    inline CELER_FUNCTION Real3 sample_displacement(Engine&   rng,
                                                    real_type cth,
                                                    real_type rmax2);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
UrbanMscScatter::UrbanMscScatter(const UrbanMscNativeRef& shared,
                                 const ParticleTrackView& particle,
                                 GeoTrackView*            geometry,
                                 const PhysicsTrackView&  physics,
                                 const MaterialView&      material,
                                 const StepLimitResult&   input)
    : inc_energy_(particle.energy())
    , inc_direction_(geometry->dir())
    , is_positron_(particle.particle_id() == shared.positron_id)
    , rad_length_(material.radiation_length())
    , mass_(shared.electron_mass.value())
    , params_(shared.params)
    , msc_(shared.msc_data[material.material_id()])
    , helper_(shared, particle, physics, material)
    , input_(input)
    , geometry_(*geometry)
{
    CELER_EXPECT(particle.particle_id() == shared.electron_id
                 || particle.particle_id() == shared.positron_id);

    lambda_ = helper_.msc_mfp(inc_energy_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the angular distribution and the lateral displacement by multiple
 * scattering as well as convert the geometrical path length to the true path
 * length based on G4VMultipleScattering::AlongStepDoIt and
 * G4UrbanMscModel::SampleScattering of the Geant4 10.7 release.
 */
template<class Engine>
CELER_FUNCTION auto UrbanMscScatter::operator()(Engine& rng)
    -> MscScatterResult
{
    // Convert the geometry path length to the true path length
    real_type geom_path = input_.geom_path;

    real_type true_path
        = helper_.calc_true_path(input_.true_path, geom_path, input_.alpha);

    // Protect against a wrong true -> geom -> true transformation
    true_path = min<real_type>(true_path, input_.phys_step);

    MscScatterResult result{true_path, inc_direction_, {0, 0, 0}};

    // Do not sample scattering at the last or at a small step
    if (true_path < helper_.range() && true_path > geom_min())
    {
        auto end_energy = helper_.end_energy(true_path);

        bool skip_sampling = (end_energy.value() < 1e-9
                              || true_path <= input_.limit_min
                              || true_path < lambda_ * params_.tau_small);

        if (!skip_sampling)
        {
            real_type cth = this->sample_cos_theta(
                rng, end_energy, input_.true_path, input_.limit_min);

            CELER_ENSURE(std::abs(cth) <= 1);

            // Sample direction
            UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
            real_type                          phi = sample_phi(rng);
            result.direction = rotate(from_spherical(cth, phi), inc_direction_);

            // Sample displacement
            real_type rmax2 = (true_path - geom_path) * (true_path + geom_path);

            if (input_.is_displaced && tau_ >= params_.tau_small && rmax2 > 0)
            {
                result.displacement
                    = this->sample_displacement(rng, phi, rmax2);
            }
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sample the scattering angle, \f$\theta\f$ at the end of the true step
 * length, \f$t\f$, described in G4UrbanMscModel::SampleCosineTheta of the
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
 * \f$g_1(u) = c_1 e^{-a(1-u)}, g_2(u) = \frac{c_2}{(b-u)^d}, g_3(u) = c_3\f$
 * with normalization constants, \f$d\f$. For small angles, \f$g_1\f$ is
 * nearly Gaussian, \f$\exp(-\frac{\theta^{2}}{2\theta_{0}^{2}})\f$, if
 * \f$\theta_0 \approx 1/1\f$, while \f$g_2\f$ has a Rutherford-like tail
 * for large $\theta$, if \f$b \approx 1\f$ and \f$d\f$ is not far from 2.
 */
template<class Engine>
CELER_FUNCTION real_type UrbanMscScatter::sample_cos_theta(Engine& rng,
                                                           Energy  end_energy,
                                                           real_type true_path,
                                                           real_type limit_min)
{
    real_type result = 1.0;

    real_type lambda_end = helper_.msc_mfp(end_energy);

    tau_ = true_path
           / ((std::fabs(lambda_ - lambda_end) > lambda_ * 0.01)
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
        real_type xmean  = std::exp(-tau_);
        real_type x2mean = (1 + 2 * std::exp(real_type(-2.5) * tau_)) / 3;

        // Too large step of the low energy particle
        if (end_energy.value() < real_type(0.5) * inc_energy_.value())
        {
            return this->simple_scattering(rng, xmean, x2mean);
        }

        // Check for extreme small steps
        real_type tsmall     = min<real_type>(limit_min, params_.lambda_limit);
        bool      small_step = (true_path < tsmall);

        real_type theta0 = (small_step)
                               ? std::sqrt(true_path / tsmall)
                                     * this->compute_theta0(tsmall, end_energy)
                               : this->compute_theta0(true_path, end_energy);

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
            real_type sth = 2 * std::sin(real_type(0.5) * theta0);
            x             = ipow<2>(sth);
        }

        // Evaluate parameters for the tail distribution
        real_type u   = small_step ? std::exp(std::log(tsmall / lambda_) / 6)
                                   : std::exp(std::log(tau_) / 6);
        real_type xsi = msc_.d[0] + u * (msc_.d[1] + msc_.d[2] * u)
                        + msc_.d[3]
                              * std::log(true_path / (tau_ * rad_length_));

        // The tail should not be too big
        xsi = max<real_type>(xsi, real_type(1.9));

        real_type c = xsi;

        if (std::abs(c - 3) < real_type(0.001))
        {
            c = real_type(3.001);
        }
        else if (std::abs(c - 2) < real_type(0.001))
        {
            c = real_type(2.001);
        }

        real_type c1 = c - 1;

        real_type ea  = std::exp(-xsi);
        real_type eaa = 1 - ea;
        // Mean of cos\theta computed from the distribution g_1(cos\theta)
        real_type xmean1 = 1 - (1 - (1 + xsi) * ea) * x / eaa;

        if (xmean1 <= real_type(0.999) * xmean)
        {
            result = this->simple_scattering(rng, xmean, x2mean);
        }

        // From continuity of derivatives
        real_type b1 = 2 + (c - xsi) * x;
        real_type bx = c * x;
        real_type d  = std::exp(std::log(bx / b1) * c1);
        real_type x0 = 1 - xsi * x;

        // Mean of cos\theta computed from the distribution g_2(cos\theta)
        real_type xmean2 = (x0 + d - (bx - b1 * d) / (c - 2)) / (1 - d);

        real_type f1x0 = ea / eaa;
        real_type f2x0 = c1 / (c * (1 - d));
        real_type prob = f2x0 / (f1x0 + f2x0);

        // Eq. 8.14 in the PRM: note that can be greater than 1
        real_type qprob = xmean / (prob * xmean1 + (1 - prob) * xmean2);

        // Sampling of cos(theta)
        if (generate_canonical(rng) < qprob)
        {
            real_type var = 0;
            if (BernoulliDistribution(prob)(rng))
            {
                // Sample \f$ \cos\theta \f$ from \f$ g_1(\cos\theta) \f$
                result = 1 + std::log(ea + generate_canonical(rng) * eaa) * x;
            }
            else
            {
                // Sample \f$ \cos\theta \f$ from \f$ g_2(\cos\theta) \f$
                var = (1 - d) * generate_canonical(rng);
                if (var < real_type(0.01) * d)
                {
                    var /= (d * c1);
                    result = -1
                             + var * (1 - real_type(0.5) * var * c)
                                   * (2 + (c - xsi) * x);
                }
                else
                {
                    result
                        = x * (c - xsi - c * std::exp(-std::log(var + d) / c1))
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
 * @param xmean the mean of \f$\cos\theta\f$.
 * @param x2mean the mean of \f$\cos\theta^{2}\f$.
 */
template<class Engine>
CELER_FUNCTION real_type UrbanMscScatter::simple_scattering(
    Engine& rng, real_type xmean, real_type x2mean) const
{
    real_type a = (2 * xmean + 9 * x2mean - 3) / (2 * xmean - 3 * x2mean + 1);
    real_type prob = (a + 2) * xmean / a;

    // Sample cos(theta)
    real_type result{};
    do
    {
        real_type rdm = generate_canonical(rng);
        result        = BernoulliDistribution(prob)(rng)
                            ? -1 + 2 * std::exp(std::log(rdm) / (a + 1))
                            : -1 + 2 * rdm;
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
 * @param true_path the true step length.
 * @param end_energy the particle energy at the end of of the msc step.
 */
CELER_FUNCTION
real_type
UrbanMscScatter::compute_theta0(real_type true_path, Energy end_energy) const
{
    real_type energy     = end_energy.value();
    real_type inc_energy = inc_energy_.value();

    real_type invbetacp = std::sqrt((inc_energy + mass_) * (energy + mass_)
                                    / (inc_energy * (inc_energy + 2 * mass_)
                                       * energy * (energy + 2 * mass_)));
    real_type y         = true_path / rad_length_;

    // Correction for the positron
    if (is_positron_)
    {
        real_type tau = std::sqrt(inc_energy * energy) / mass_;
        y *= this->calc_correction(tau);
    }

    // Note: multiply abs(charge) if the charge number is not unity
    real_type theta0 = c_highland().value() * std::sqrt(y) * invbetacp;

    // Correction factor from e- scattering data
    theta0 *= (msc_.coeffth1 + msc_.coeffth2 * std::log(y));

    return theta0;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the correction on theta0 for positrons.
 *
 * @param tau (incident energy * energy at the end of step)/electron_mass.
 */
CELER_FUNCTION real_type UrbanMscScatter::calc_correction(real_type tau) const
{
    real_type corr{1.0};

    real_type           zeff = msc_.zeff;
    constexpr real_type xl   = 0.6;
    constexpr real_type xh   = 0.9;
    constexpr real_type e    = 113;

    real_type x = std::sqrt(tau * (tau + 2) / ipow<2>(tau + 1));
    real_type a = real_type(0.994) - real_type(4.08e-3) * zeff;
    real_type b = real_type(7.16) + (real_type(52.6) + 365 / zeff) / zeff;
    real_type c = 1 - real_type(4.47e-3) * zeff;
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
        corr         = y0 * x + y1;
    }
    corr *= (zeff * (real_type(1.84035e-4) * zeff - real_type(1.86427e-2))
             + real_type(1.41125));

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
 * @param phi the azimuthal angle of the multiple scattering.
 * @param rmax2 the asymmetry between true path and geom path
 */
template<class Engine>
CELER_FUNCTION Real3 UrbanMscScatter::sample_displacement(Engine&   rng,
                                                          real_type phi,
                                                          real_type rmax2)
{
    // Sample a unit direction of the displacement
    constexpr real_type cbeta  = 2.160;
    real_type           cbeta1 = 1 - std::exp(-cbeta * constants::pi);

    real_type psi   = -std::log(1 - generate_canonical(rng) * cbeta1) / cbeta;
    real_type angle = BernoulliDistribution(0.5)(rng) ? phi + psi : phi - psi;

    Real3 displacement{std::cos(angle), std::sin(angle), 0};

    // Rotate along the incident particle direction
    displacement = rotate(displacement, inc_direction_);

    // Scale with the lateral arm
    real_type arm = real_type(0.73) * std::sqrt(rmax2);
    for (auto i : range(3))
    {
        displacement[i] *= arm;
    }

    // Do not sample near the boundary
    real_type rho = norm(displacement);
    if (rho > params_.geom_limit)
    {
        real_type safety = (1 - params_.safety_tol)
                           * geometry_.find_safety(geometry_.pos());
        real_type multiplier = 0;

        if (rho <= safety)
        {
            multiplier = 1;
        }
        else if (safety > params_.geom_limit)
        {
            multiplier = safety / rho;
        }
        // Otherwise (near a volume boundary), do not change position

        for (auto i : range(3))
        {
            displacement[i] *= multiplier;
        }
    }

    return displacement;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
