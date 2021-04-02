//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SBEnergyDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Units.hh"
#include "physics/grid/TwodSubgridCalculator.hh"
#include "random/distributions/UniformRealDistribution.hh"
#include "SeltzerBerger.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample exiting photon energy from Bremsstrahlung.
 *
 * The SB energy distribution uses tabulated Seltzer-Berger differential cross
 * section data from G4EMLOW, which stores scaled cross sections as a function
 * of incident particle energy and exiting gamma energy (see SeltzerBergerModel
 * for details). The sampling procedure is roughly laid out in section
 * [PHYS341] of the GEANT3 physics reference manual, although like Geant4 we
 * use raw tabulated SB data rather than a parameter fit. Also like Geant4 we
 * include the extra density correction factor.
 *
 * Once an element and incident energy have been selected, the exiting energy
 * distribution is the differential cross section, which is stored as a scaled
 * tabulated value. The reconstructed cross section gives the pdf
 * \f[
 *   p(\kappa) \propto \frac{d \sigma}{dk}s
               \propto \frac{1}{\kappa} \chi_Z(E, \kappa)
 * \f]
 * where \f$ \kappa = k / E \f$ is the ratio of the emitted photon energy to
 * the incident charged particle energy, and the domain of \em p is
 * nominally restricted by the allowable energy range \f$ \kappa_c < \kappa < 1
 * \f$, where \f$ \kappa_c \f$ is from the cutoff energy \f$ E_c \f$ for
 * secondary gamma production. This distribution is sampled by decomposing it
 * into an analytic sampling of \f$ 1/\kappa \f$ and a rejection sample on
 * the scaled cross section \f$ \chi \f$.
 * The analytic sample over the domain is from \f[
   p_1(\kappa) \frac{1}{\ln 1/\kappa_c} \frac{1}{\kappa}
 * \f]
 * by sampling
 * \f[
   \kappa = \exp( \xi \ln \kappa_c ) \,,
   \f]
 * and the rejection sample is the ratio of the scaled differential cross
 * section at the exiting energy to the maximum across all exiting energies.
 * \f[
   p_2(\kappa) = \frac{\chi_Z(E, \kappa)}{\max_\kappa \chi_Z(E, \kappa)}
   \f]
 * Since the tabulated data is bilinearly interpolated in incident and exiting
 * energy, we can calculate a bounding maximum by precalculating (at
 * setup time) the index of the maximum value of \f$ \chi \f$ and
 * linearly interpolating those maximum values based on the incident energy.
 *
 * The above algorithm is idealized; in practice, the minimum and maximum
 * values are adjusted for a constant factor \f$d_\rho\f$, which depends on the
 * incident particle mass + energy and the material's electron density:
 * \f[
    \kappa_\mathrm{min} = \ln (E_c^2 + d_\rho E^2)
 * \f]
 * and
 * \f[
    \kappa_\mathrm{max} = \ln (E^2 + d_\rho E^2) \, .
 * \f]
 * With this correction, the sample of the exiting gamma energy
 * \f$ k = \kappa / E \f$ is transformed from the simple exponential above to:
 * \f[
 *  k = \sqrt{ \exp(\kappa_\mathrm{min} + \xi [\kappa_\mathrm{max} -
 \kappa_\mathrm{min}]) - d_\rho E^2}
 * \f]
 *
 * \todo Add a template parameter for a cross section scaling, which will be an
 * identity function (always returning 1) for electrons but is
 * SBPositronXsScaling for positrons. Use it to adjust max_xs on construction
 * and modify the sampled xs during iteration.
 */
class SBEnergyDistribution
{
  public:
    //!@{
    //! Type aliases
    using SBData
        = SeltzerBergerData<Ownership::const_reference, MemSpace::native>;
    using Energy   = units::MevEnergy;
    using EnergySq = Quantity<UnitProduct<units::Mev, units::Mev>>;
    //!@}

  public:
    // Construct from data
    inline CELER_FUNCTION SBEnergyDistribution(const SBData& data,
                                               Energy        inc_energy,
                                               ElementId     element,
                                               EnergySq density_correction,
                                               Energy   min_gamma_energy);

    template<class Engine>
    inline CELER_FUNCTION Energy operator()(Engine& rng);

    //// DEBUG FUNCTIONALITY ////

    //! Maximum cross section calculated for rejection
    real_type max_xs() const { return 1 / inv_max_xs_; }

  private:
    //// IMPLEMENTATION DATA ////

    using SBTables
        = SeltzerBergerTableData<Ownership::const_reference, MemSpace::native>;
    using UniformSampler = UniformRealDistribution<>;

    const real_type             inc_energy_;
    const TwodSubgridCalculator calc_xs_;
    const real_type             inv_max_xs_;

    const real_type dens_corr_;
    UniformSampler  sample_log_exit_efrac_;

    //// CONSTRUCTION HELPER FUNCTIONS ////

    inline CELER_FUNCTION TwodSubgridCalculator
    make_xs_calc(const SBTables&, ElementId element) const;

    inline CELER_FUNCTION real_type calc_max_xs(const SBTables&,
                                                ElementId element) const;

    inline CELER_FUNCTION UniformSampler
    make_lee_sampler(real_type min_gamma_energy) const;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "SBEnergyDistribution.i.hh"
