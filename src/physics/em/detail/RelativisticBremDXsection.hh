//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RelativisticBremDXsection.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Algorithms.hh"
#include "base/Constants.hh"
#include "base/Macros.hh"
#include "base/Quantity.hh"
#include "base/Types.hh"

#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Units.hh"
#include "physics/material/Types.hh"

#include "RelativisticBremData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the different cross sections with and without the LPM effect.
 */
class RelativisticBremDXsection
{
  public:
    //!@{
    //! Type aliases
    using R           = real_type;
    using Energy      = units::MevEnergy;
    using ItemIdT     = celeritas::ItemId<unsigned int>;
    using ElementData = detail::RelBremElementData;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    RelativisticBremDXsection(const RelativisticBremNativeRef& shared,
                              const ParticleTrackView&         particle,
                              const MaterialView&              material,
                              const ElementComponentId&        elcomp_id);

    // Compute cross section
    inline CELER_FUNCTION real_type operator()(real_type energy);

    // Return the density correction
    CELER_FUNCTION real_type density_correction() const
    {
        return density_corr_;
    }

    // Return the maximum value of the differential cross section
    CELER_FUNCTION real_type maximum_value() const
    {
        return elem_data_.factor1 + elem_data_.factor2;
    }

  private:
    //// DATA ////

    // Shared constant physics properties
    const RelativisticBremNativeRef& shared_;
    // Element data of the current material
    const ElementData& elem_data_;
    // Total energy of the incident particle
    real_type total_energy_;
    // Density correction for the current material
    real_type density_corr_;
    // LPM energy for the current material
    real_type lpm_energy_;
    // Flag for the LPM effect
    bool enable_lpm_;

    //// HELPER TYPES ////

    //! Intermediate data for screening functions
    struct ScreenFunctions
    {
        real_type phi1{0};
        real_type phi2{0};
        real_type psi1{0};
        real_type psi2{0};
    };

    //! LPM functions
    struct LPMFunctions
    {
        real_type xis{0};
        real_type gs{0};
        real_type phis{0};
    };

    /// STATIC CONSTANTS ///

    // Migdal constant: 4\pi * r_0 * electron_reduced_compton_wavelength^2
    static CELER_CONSTEXPR_FUNCTION real_type migdal_constant()
    {
        using namespace constants;
        return real_type(4) * pi * r_electron * ipow<2>(lambdabar_electron);
    }

    // LPM constant: \alpha * (mc^2)^2 /(4\pi * \hbar c) in [MeV/cm]
    static CELER_CONSTEXPR_FUNCTION real_type lpm_constant()
    {
        using namespace constants;
        return alpha_fine_structure * ipow<2>(electron_mass * ipow<2>(c_light))
               / (real_type(4) * pi * hbar_planck * c_light)
               / unit_cast(units::MevEnergy{1});
    }

    //// HELPER FUNCTIONS ////

    //! Calculate the differential cross section per atom
    inline CELER_FUNCTION real_type dxsec_per_atom(real_type energy);

    //! Calculate the differential cross section per atom with the LPM effect
    inline CELER_FUNCTION real_type dxsec_per_atom_lpm(real_type energy);

    //! Compute screening functions
    inline CELER_FUNCTION ScreenFunctions
                          compute_screen_functions(real_type gamma, real_type epsilon);

    //! Compute LPM functions
    inline CELER_FUNCTION LPMFunctions compute_lpm_functions(real_type ss);
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "RelativisticBremDXsection.i.hh"
