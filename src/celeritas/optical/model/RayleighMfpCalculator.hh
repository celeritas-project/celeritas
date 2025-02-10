//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighMfpCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/optical/detail/OpticalUtils.hh"

#include "../MaterialView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the Rayleigh MFP for a given set of material properties.
 *
 * Uses the Einstein-Smoluchowski formula to calculate the mean free path
 * at a given energy. In Landau and Lifshitz Electrodynamics of Continuous
 * Media, the mean free path is given by equation (120.2):
 * \f[
    l^{-1} = \frac{1}{6\pi} k^4 \rho k_B T \left(\frac{\partial \rho}{\partial
 P}\right)_T \left(\frac{\partial \varepsilon}{\partial \rho}\right)_T^2
 * \f]
 * where we only consider density fluctations at constant temperature. The
 * first partial derivative may be rewritten in terms of the isothermal
 * compressibility \f$ \beta_T \f$:
 * \f[
    \left(\frac{\partial \rho}{\partial P}\right)_T = \rho \beta_T.
 * \f]
 * The latter partial derivative may be calculated via the Clausius-Mossetti
 * equation
 * \f[
    \frac{\varepsilon - 1}{\varepsilon + 2} = A \rho
 * \f]
 * for constant \f$ A \f$, giving
 * \f[
    \left(\frac{\partial \varepsilon}{\partial \rho}\right)_T =
 \frac{(\varepsilon - 1)(\varepsilon + 2)}{3\rho}.
 * \f]
 * The final equation for the MFP in terms of energy:
 * \f[
    l^{-1} = \frac{k_B T \beta_T}{6\pi} \left(\frac{E}{\hbar c}\right)^4 \left[
 \frac{(\varepsilon - 1)(\varepsilon + 2)}{3} \right]^2.
 * \f]
 *
 * The scale factor is a unitless user customizable factor that's multiplied
 * to the inverse MFP.
 */
class RayleighMfpCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Grid = typename NonuniformGridCalculator::Grid;
    //!@}

  public:
    // Construct from material and Rayleigh properties
    inline CELER_FUNCTION
    RayleighMfpCalculator(MaterialView const& material,
                          ImportOpticalRayleigh const& rayleigh,
                          ::celeritas::MaterialView const& core_material);

    // Calculate the MFP for the given energy
    inline CELER_FUNCTION real_type operator()(Energy) const;

    //! Retrieve the underlying energy grid used to calculate the MFP
    inline CELER_FUNCTION Grid const& grid() const
    {
        return calc_rindex_.grid();
    }

  private:
    // Calculate refractive index [MeV -> unitless]
    NonuniformGridCalculator calc_rindex_;

    // Constant prefactor at all energies
    real_type density_fluctuation_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
RayleighMfpCalculator::RayleighMfpCalculator(
    MaterialView const& material,
    ImportOpticalRayleigh const& rayleigh,
    ::celeritas::MaterialView const& core_material)
    : calc_rindex_(material.make_refractive_index_calculator())
    , density_fluctuation_(rayleigh.scale_factor * rayleigh.compressibility
                           * core_material.temperature()
                           * celeritas::constants::k_boltzmann
                           / (6 * celeritas::constants::pi))
{
    CELER_EXPECT(rayleigh);
    CELER_EXPECT(material.material_id() == core_material.optical_material_id());
    CELER_EXPECT(density_fluctuation_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the optical Rayleigh mean free path at the given energy.
 */
CELER_FUNCTION real_type RayleighMfpCalculator::operator()(Energy energy) const
{
    CELER_EXPECT(energy > zero_quantity());

    real_type rindex = calc_rindex_(value_as<Energy>(energy));
    CELER_ASSERT(rindex > 1);

    real_type wave_number = 2 * celeritas::constants::pi
                            / detail::energy_to_wavelength(energy);
    real_type permitivity_fluctuation = (ipow<2>(rindex) - 1)
                                        * (ipow<2>(rindex) + 2) / 3;

    return 1
           / (density_fluctuation_ * ipow<4>(wave_number)
              * ipow<2>(permitivity_fluctuation));
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
