//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/TsaiUrbanDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample exiting polar angle for pair production and bremsstrahlung.
 *
 * For pair production, the polar angle of the electron (or positron) is
 * defined with respect to the direction of the parent photon. The energy-
 * angle distribution given by Tsai is quite complicated to
 * sample and can be approximated by a density function suggested by Urban.
 *
 * The angular distribution of the emitted photons is obtained from a
 * simplified formula based on the Tsai cross-section,
 * which is expected to become isotropic in the low energy limit.
 *
 * The sample result is the cosine of the exiting angle with respect to the
 * incident angle.
 *
 * \note This performs the same sampling routine as in Geant4's
 * ModifiedTsai class, based on derivation from \cite{tsai-1974}
 * and documented in section 6.5.2 (pair-production), and 10.2.1 and 10.2.4
 * (bremsstrahlung) of the Geant4 Physics Reference \cite{g4prm}
 * (release 10.6).
 */
class TsaiUrbanDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using result_type = real_type;
    //!@}

  public:
    // Construct with defaults
    inline CELER_FUNCTION TsaiUrbanDistribution(Energy energy, Mass mass);

    // Sample cos(theta) using the given random number generator
    template<class Engine>
    inline CELER_FUNCTION result_type operator()(Engine& rng);

  private:
    // Dimensionless ratio of energy [Mev] to  mass * c^2 [MevMass*c^2]
    real_type umax_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from input data.
 */
CELER_FUNCTION
TsaiUrbanDistribution::TsaiUrbanDistribution(Energy energy, Mass mass)
    : umax_(2 * (1 + energy.value() / mass.value()))
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample the cosine of the polar angle of the exiting gamma.
 */
template<class Engine>
CELER_FUNCTION real_type TsaiUrbanDistribution::operator()(Engine& rng)
{
    real_type u;
    do
    {
        real_type uu
            = -std::log(generate_canonical(rng) * generate_canonical(rng));
        u = uu
            * (BernoulliDistribution(0.25)(rng) ? real_type(1.6)
                                                : real_type(1.6 / 3));
    } while (u > umax_);

    return 1 - 2 * ipow<2>(u / umax_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
