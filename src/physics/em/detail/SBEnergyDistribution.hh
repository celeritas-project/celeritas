//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SBEnergyDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample exiting photon energy from Bremsstrahlung.
 *
 * The SB energy distribution uses tabulated Seltzer-Berger differential cross
 * section data, which stores cumulative probabilities as a function of
 * incident particle energy and exiting gamma energy (see SeltzerBergerModel
 * for details). The sampling procedure is roughly laid out in section
 * [PHYS341] of the GEANT3 physics reference manual, although like Geant4 we
 * use raw tabulated  SB data rather than a parameter fit. Also like Geant4 we
 * include the extra density correction factor.
 *
 * The algorithm is roughly thus. Calculate allowable exit energy range from
 * the gamma energy production cutoff and the incident particle energy. The
 * exiting energy fraction \em x of the kinetic energy transferred to the
 * photon is nominally between \em x_c and 1, where \em x_c is from
 * the cutoff energy for producing gammas \em E_c. However due to a correction
 * factor \f$k_\rho\f$ (which depends on the incident particle mass + energy
 * and the material's electron density) the exiting energy cutoffs are
 *
 * The minimum and maximum values of \em x, adjusted for the correction factor,
 * are:
 * \f[
    x_\mathrm{min} = \ln (E_c^2 + k_\rho E^2)
 * \f]
 * and
 * \f[
    x_\mathrm{max} = \ln (E + k_\rho E^2)
 * \f]
 *
 * The exiting kinetic energy is calculated in a rejection loop: without the
 * correction factor, the provisional value of \em x would be sampled from
 * \f[
   \frac{1}{\ln 1/x_c} \frac{1}{x}
 * \f]
 * by sampling
 * \f[
   x = \exp( \xi \ln x_c )
   \f]
 * but with a nonzero correction the exiting photon kinetic energy is sampled
 * from
 * \f[
 *  E_\gamma = \sqrt{ \exp(x_\mathrm{min} + \xi [x_\mathrm{max} -
 x_\mathrm{min}]) - k_\rho E^2}
 * \f]
 *
 * The acceptance condition for the loop is the ratio of the cross differential
 * cross section \f$ S(\ln E, x) \f$ to the maximum cross section \f$
 * S_\mathrm{max}(\ln E)\f$. These two values are looked up via the
 * precalculated 2D Seltzer-Berger tables imported from G4EMLOW data. The
 * index of the maximum value of \em S at each energy grid point is stored in
 * advance -- this allows us to efficiently calculate the exactly correct
 * maximum through linear interpolation for a given energy.
 */
class SBEnergyDistribution
{
  public:
    //!@{
    //! Type aliases
    using SBData
        = SeltzerBergerData<Ownership::const_reference, MemSpace::native>;
    using EnergySq = real_type;
    //!@}

  public:
    // Construct from data
    inline CELER_FUNCTION
    SBEnergyDistribution(const SBData&            data,
                         const ParticleTrackView& particle,
                         ElementId                element,
                         EnergySq                 density_correction,
                         Energy                   min_gamma_energy);

    template<class Engine>
    inline CELER_FUNCTION Energy operator()(Engine& rng);

  private:
    //// IMPLEMENTATION TYPES ////

    enum class ParticleType : bool
    {
        positron,
        electron
    };

    //// IMPLEMENTATION DATA ////

    const SBData&      shared_;
    const ParticleType pt_;
    real_type          inc_energy_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "SBEnergyDistribution.i.hh"
