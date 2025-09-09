//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/MuAngularDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample the polar angle for muon bremsstrahlung and pair production.
 *
 * The polar angle is sampled according to a simplified PDF
 * \f[
   f(r) \sim \frac{r}{(1 + r^2)^2}, \ r = frac{E\theta}{m}
 * \f]
 * by sampling
 * \f[
   \theta = r\frac{m}{E}
 * \f]
 * with
 * \f[
   r = \sqrt{frac{a}{1 - a}},
   a = \xi \frac{r^2_{\text{max}}}{1 + r^2_{\text{max}}},
   r_{\text{max}} = frac{\pi}{2} E' / m \min(1, E' / \epsilon),
 * \f]
 * and where \f$ m \f$ is the incident muon mass, \f$ E \f$ is incident energy,
 * \f$ \epsilon \f$ is the emitted energy,
 * \f$ E' = E - \epsilon \f$, and \f$ \xi \sim U(0,1) \f$.
 *
 * \note This performs the same sampling routine as in Geant4's \c
 * G4ModifiedMephi class and documented in section 11.2.4 of the Geant4 Physics
 * Reference (release 11.2).
 */
class MuAngularDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    //!@}

  public:
    // Construct with incident and secondary particle quantities
    inline CELER_FUNCTION
    MuAngularDistribution(Energy inc_energy, Mass inc_mass, Energy energy);

    // Sample the cosine of the polar angle of the secondary
    template<class Engine>
    inline CELER_FUNCTION real_type operator()(Engine& rng);

  private:
    // Incident particle Lorentz factor
    real_type gamma_;
    // r_max^2 / (1 + r_max^2)
    UniformRealDistribution<real_type> sample_a_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with incident and secondary particle.
 */
CELER_FUNCTION
MuAngularDistribution::MuAngularDistribution(Energy inc_energy,
                                             Mass inc_mass,
                                             Energy energy)
    : gamma_(1 + value_as<Energy>(inc_energy) / value_as<Mass>(inc_mass))
{
    real_type r_max_sq = ipow<2>(
        real_type(0.5) * constants::pi * gamma_
        * min(real_type(1),
              gamma_ * value_as<Mass>(inc_mass) / value_as<Energy>(energy) - 1));
    sample_a_ = {0, r_max_sq / (1 + r_max_sq)};
}

//---------------------------------------------------------------------------//
/*!
 * Sample the cosine of the polar angle of the secondary.
 */
template<class Engine>
CELER_FUNCTION real_type MuAngularDistribution::operator()(Engine& rng)
{
    real_type a = sample_a_(rng);
    return std::cos(std::sqrt(a / (1 - a)) / gamma_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
