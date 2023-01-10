//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/detail/PhysicsConstants.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Constants.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Physical constants which are derived from fundamental constants.
 *
 * Derived constants    | Unit                  | Notes
 * -------------------- | --------------------- | ------------
 * electron_mass_c2()   | g * (cm/s)^2          |
 * migdal_constant()    | cm^3                  | Bremsstrahlung
 * lpm_constant()       | Mev/cm                | Relativistic Bremsstrahlung
 */

//!@{
//! Type aliases
using MevPerCm = Quantity<UnitDivide<units::Mev, units::NativeUnit>>;
//!@}

//!@{
//! Constant functions
CELER_CONSTEXPR_FUNCTION real_type electron_mass_c2()
{
    using namespace constants;

    return electron_mass * c_light * c_light;
}

CELER_CONSTEXPR_FUNCTION real_type migdal_constant()
{
    using namespace constants;
    using namespace units;

    return 4 * pi * r_electron * lambdabar_electron * lambdabar_electron;
}

CELER_CONSTEXPR_FUNCTION MevPerCm lpm_constant()
{
    using namespace constants;
    using namespace units;

    // This is used to calculate the LPM characteristic energy, defined as \f$
    // E_\textrm{LPM} = \frac{\alpha m^2 X_0}{2 h c} \f$, where \f$ X_0 \f$ is
    // the radiation length of the material. Note that some papers define \f$
    // E_\textrm{LPM} \f$ as a factor of two smaller and others as a factor of
    // 8 larger (see S. Klein, Suppression of bremsstrahlung and pair
    // production due to environmental factors, Rev. Mod. Phys. 71 (1999)
    // 1501-1538). The Geant4 Physicss Reference Manual (Eq. 10.17) has an
    // extra factor of two in the denominator.
    return native_value_to<MevPerCm>(alpha_fine_structure * electron_mass_c2()
                                     * electron_mass_c2()
                                     / (2 * h_planck * c_light));
}
//!@}

//---------------------------------------------------------------------------//
// Constant functions for model limits
//---------------------------------------------------------------------------//

//!@{
//! Maximum energy for the SeltzerBerger model - TODO: make this configurable
CELER_CONSTEXPR_FUNCTION units::MevEnergy seltzer_berger_limit()
{
    return units::MevEnergy{1e3};  //! 1 GeV
}

//! Maximum energy for EM models to be valid
CELER_CONSTEXPR_FUNCTION units::MevEnergy high_energy_limit()
{
    return units::MevEnergy{1e8};  //! 100 TeV
}
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
