//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/SBEnergyDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/grid/TwodSubgridCalculator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/SeltzerBergerData.hh"

#include "SBEnergyDistHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Default scaling for SB cross sections.
 */
struct SBElectronXsCorrector
{
    using Xs = RealQuantity<SBElementTableData::XsUnits>;

    //! No cross section scaling for any exiting energy
    CELER_FUNCTION real_type operator()(units::MevEnergy) const { return 1; }
};

//---------------------------------------------------------------------------//
/*!
 * Sample exiting photon energy from Bremsstrahlung.
 *
 * The SB energy distribution uses tabulated Seltzer-Berger differential cross
 * section data from G4EMLOW, which stores scaled cross sections as a function
 * of incident particle energy and exiting gamma energy (see SeltzerBergerModel
 * for details). The sampling procedure is roughly laid out in section
 * `[PHYS341]` of the GEANT3 reference manual \citep{geant3-1993,
 * https://cds.cern.ch/record/1082634} , although like Geant4 we
 * use raw tabulated SB data rather than a parameter fit. Also like Geant4 we
 * include the extra density correction factor.
 *
 * Once an element and incident energy have been selected, the exiting energy
 * distribution is the differential cross section, which is stored as a scaled
 * tabulated value. The reconstructed cross section gives the pdf
 * \f[
 *   p(\kappa) \propto \difd{\sigma}{k} s
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
 * Most of the mechanics of the sampling are in the template-free
 * \c SBEnergyDistHelper, which is passed as a construction argument to this
 * sampler. The separate class exists here to minimize duplication of templated
 * code, which is required to provide for an on-the-fly correction of the cross
 * section sampling.
 */
template<class XSCorrector>
class SBEnergyDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using SBData = NativeCRef<SeltzerBergerData>;
    using Energy = units::MevEnergy;
    using EnergySq = RealQuantity<UnitProduct<units::Mev, units::Mev>>;
    //!@}

  public:
    // Construct from data
    inline CELER_FUNCTION SBEnergyDistribution(SBEnergyDistHelper const& helper,
                                               XSCorrector scale_xs);

    template<class Engine>
    inline CELER_FUNCTION Energy operator()(Engine& rng);

  private:
    //// IMPLEMENTATION DATA ////
    SBEnergyDistHelper const& helper_;
    XSCorrector scale_xs_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from incident particle and energy.
 *
 * The incident energy *must* be within the bounds of the SB table data, so the
 * Model's applicability must be consistent with the table data.
 */
template<class X>
CELER_FUNCTION
SBEnergyDistribution<X>::SBEnergyDistribution(SBEnergyDistHelper const& helper,
                                              X scale_xs)
    : helper_(helper), scale_xs_(::celeritas::move(scale_xs))
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample the exiting energy by doing a table lookup and rejection.
 */
template<class X>
template<class Engine>
CELER_FUNCTION auto SBEnergyDistribution<X>::operator()(Engine& rng) -> Energy
{
    // Sampled energy
    Energy exit_energy;
    // Calculated cross section used inside rejection sampling
    real_type xs{};
    do
    {
        // Sample scaled energy and subtract correction factor
        exit_energy = helper_.sample_exit_energy(rng);

        // Interpolate the differential cross setion at the sampled exit energy
        xs = helper_.calc_xs(exit_energy).value() * scale_xs_(exit_energy);
    } while (RejectionSampler<>(xs, helper_.max_xs().value())(rng));
    return exit_energy;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
