//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsConstants.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Constants.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
namespace detail
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
    return units::MevEnergy{1e3}; //! 1 GeV
}

//! Maximum energy for EM models to be valid
CELER_CONSTEXPR_FUNCTION units::MevEnergy high_energy_limit()
{
    return units::MevEnergy{1e8}; //! 100 TeV
}
//!@}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
