//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/RBDiffXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/PolyEvaluator.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/RelativisticBremData.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "LPMCalculator.hh"
#include "ScreeningFunctions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate differential cross sections for relativistic bremsstrahlung.
 *
 * This accounts for the LPM effect if the option is enabled and the
 * electron energy is high enough.
 *
 * The screening functions are documented in \c
 * celeritas::TsaiScreeningCalculator.
 *
 * \note This is currently used only as a shape function for rejection, so as
 * long as the resulting cross section is scaled by the maximum value the units
 * do not matter.
 */
class RBDiffXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using ElementData = RelBremElementData;
    //!@}

  public:
    // Construct with incident electron and current element
    inline CELER_FUNCTION RBDiffXsCalculator(RelativisticBremRef const& shared,
                                             ParticleTrackView const& particle,
                                             MaterialView const& material,
                                             ElementComponentId elcomp_id);

    // Compute cross section of exiting gamma energy
    inline CELER_FUNCTION real_type operator()(Energy energy);

    //! Density correction factor [Energy^2]
    CELER_FUNCTION real_type density_correction() const
    {
        return density_corr_;
    }

    //! Return the maximum value of the differential cross section
    CELER_FUNCTION real_type maximum_value() const
    {
        return elem_data_.factor1 + elem_data_.factor2;
    }

  private:
    //// TYPES ////

    using R = real_type;

    //// DATA ////

    // Element data of the current material
    ElementData const& elem_data_;
    // Shared problem data for the current material
    MaterialView const& material_;
    // Shared problem data for the current element
    ElementView const element_;
    // Total energy of the incident particle
    real_type total_energy_;
    // Density correction for the current material
    real_type density_corr_;
    // Flag for the LPM effect
    bool enable_lpm_;
    // Flag for dialectric suppression effect in LPM
    bool dielectric_suppression_;

    //// HELPER FUNCTIONS ////

    //! Calculate the differential cross section per atom
    inline CELER_FUNCTION real_type dxsec_per_atom(real_type energy);

    //! Calculate the differential cross section per atom with the LPM effect
    inline CELER_FUNCTION real_type dxsec_per_atom_lpm(real_type energy);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with incident electron and current element.
 */
CELER_FUNCTION
RBDiffXsCalculator::RBDiffXsCalculator(RelativisticBremRef const& shared,
                                       ParticleTrackView const& particle,
                                       MaterialView const& material,
                                       ElementComponentId elcomp_id)
    : elem_data_(shared.elem_data[material.element_id(elcomp_id)])
    , material_(material)
    , element_(material.element_record(elcomp_id))
    , total_energy_(value_as<Energy>(particle.total_energy()))
    , dielectric_suppression_(shared.dielectric_suppression())
{
    real_type density_factor = material.electron_density()
                               * detail::migdal_constant();
    density_corr_ = density_factor * ipow<2>(total_energy_);

    real_type lpm_energy
        = material.radiation_length()
          * value_as<detail::MevPerLen>(detail::lpm_constant());
    real_type lpm_threshold = lpm_energy * std::sqrt(density_factor);
    enable_lpm_ = (shared.enable_lpm && (total_energy_ > lpm_threshold));
}

//---------------------------------------------------------------------------//
/*!
 * Compute the relativistic differential cross section per atom at the given
 * bremsstrahlung photon energy in MeV.
 */
CELER_FUNCTION
real_type RBDiffXsCalculator::operator()(Energy energy)
{
    CELER_EXPECT(energy > zero_quantity());
    return enable_lpm_ ? this->dxsec_per_atom_lpm(energy.value())
                       : this->dxsec_per_atom(energy.value());
}

//---------------------------------------------------------------------------//
/*!
 * Compute the differential cross section without the LPM effect.
 */
CELER_FUNCTION
real_type RBDiffXsCalculator::dxsec_per_atom(real_type gamma_energy)
{
    real_type dxsec{0};

    real_type y = gamma_energy / total_energy_;
    real_type term0 = PolyEvaluator<real_type, 2>{1.0, -1.0, 0.75}(y);

    if (element_.atomic_number() < AtomicNumber{5})
    {
        // The Dirac-Fock model
        dxsec = term0 * elem_data_.factor1 + (1 - y) * elem_data_.factor2;
    }
    else
    {
        // Tsai's analytical approximation
        using Mass = ElementData::Mass;
        using InvEnergy = RealQuantity<UnitInverse<Energy::unit_type>>;
        auto sfunc = TsaiScreeningCalculator{Mass{elem_data_.gamma_factor},
                                             Mass{elem_data_.epsilon_factor}}(
            InvEnergy{y / (total_energy_ - gamma_energy)});

        real_type invz = 1
                         / static_cast<real_type>(
                             element_.atomic_number().unchecked_get());
        dxsec = term0
                    * ((R(0.25) * sfunc.phi1 - elem_data_.fz)
                       + (R(0.25) * sfunc.psi1 - 2 * element_.log_z() / 3)
                             * invz)
                + R(0.125) * (1 - y) * (sfunc.dphi + sfunc.dpsi * invz);
    }

    return clamp_to_nonneg(dxsec);
}

//---------------------------------------------------------------------------//
/*!
 * Compute the differential cross section with the LPM effect.
 */
CELER_FUNCTION
real_type RBDiffXsCalculator::dxsec_per_atom_lpm(real_type gamma_energy)
{
    // Evaluate LPM functions
    real_type epsilon = total_energy_ / gamma_energy;
    LPMCalculator calc_lpm_functions(
        material_, element_, dielectric_suppression_, Energy{gamma_energy});
    auto lpm = calc_lpm_functions(epsilon);

    real_type y = gamma_energy / total_energy_;
    real_type hy_sq = R(0.25) * ipow<2>(y);
    real_type term = lpm.xi * (hy_sq * lpm.g + (1 - y + 2 * hy_sq) * lpm.phi);

    real_type dxsec = term * elem_data_.factor1 + (1 - y) * elem_data_.factor2;

    return clamp_to_nonneg(dxsec);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
