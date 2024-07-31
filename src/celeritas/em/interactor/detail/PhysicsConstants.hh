//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/detail/PhysicsConstants.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/UnitUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/UnitTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Special partly-natural unit [MeV / len]
using MevPerLen = Quantity<UnitDivide<units::Mev, units::Native>>;

//! Migdal's constant used for Bremsstrahlung [len^3]
CELER_CONSTEXPR_FUNCTION real_type migdal_constant()
{
    using namespace constants;

    return 4 * pi * r_electron * ipow<2>(lambdabar_electron);
}

//---------------------------------------------------------------------------//
/*!
 * Landau-Pomeranchuk-Migdal constant [MeV / len].
 *
 * This is used to calculate the LPM characteristic energy, defined as
 * \f$ E_\textrm{LPM} = \frac{\alpha m^2 X_0}{2 h c} \f$, where
 * \f$ X_0 \f$ is the radiation length of the material. Note that some papers
 * define \f$ E_\textrm{LPM} \f$ as a factor of two smaller and others as a
 * factor of 8 larger (see S. Klein, Suppression of bremsstrahlung and pair
 * production due to environmental factors, Rev. Mod. Phys. 71 (1999)
 * 1501-1538). The Geant4 Physics Reference Manual (Eq. 10.17) has an
 * extra factor of two in the denominator.
 */
CELER_CONSTEXPR_FUNCTION MevPerLen lpm_constant()
{
    using namespace constants;

    constexpr real_type electron_mass_csq = electron_mass * ipow<2>(c_light);

    return native_value_to<MevPerLen>(alpha_fine_structure
                                      * ipow<2>(electron_mass_csq)
                                      / (2 * h_planck * c_light));
}

//---------------------------------------------------------------------------//
// Constant functions for model limits
//---------------------------------------------------------------------------//

//! Maximum energy for the SeltzerBerger model - TODO: make this configurable
CELER_CONSTEXPR_FUNCTION units::MevEnergy seltzer_berger_upper_limit()
{
    return units::MevEnergy{1e3};  //! 1 GeV
}

//! Energy threshold between muon ionization models. TODO: make configurable
CELER_CONSTEXPR_FUNCTION units::MevEnergy mu_bethe_bloch_lower_limit()
{
    return units::MevEnergy{0.2};  //! 200 keV
}

//! Maximum energy for EM models to be valid
CELER_CONSTEXPR_FUNCTION units::MevEnergy high_energy_limit()
{
    return units::MevEnergy{1e8};  //! 100 TeV
}

//! Used in Bragg model to calculate minimum energy transfer to delta ray
CELER_CONSTEXPR_FUNCTION units::MevEnergy bragg_lowest_kin_energy()
{
    return units::MevEnergy{2.5e-4};  //! 0.25 keV
}

//! Used in ICRU73QO model to calculate minimum energy transfer to delta ray
CELER_CONSTEXPR_FUNCTION units::MevEnergy icru73qo_lowest_kin_energy()
{
    return units::MevEnergy{5e-3};  //! 5 keV
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
