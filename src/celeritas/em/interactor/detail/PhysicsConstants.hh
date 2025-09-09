//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/detail/PhysicsConstants.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Constant.hh"
#include "corecel/math/UnitUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/UnitTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Special partly-natural unit [MeV / len]
using MevPerLen = RealQuantity<UnitDivide<units::Mev, units::Native>>;

//! Migdal's constant used for Bremsstrahlung [len^3]
CELER_CONSTEXPR_FUNCTION Constant migdal_constant()
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
 * factor of 8 larger: see \cite{klein-lpm-1999}.
 * The Geant4 Physics Reference Manual (Eq. 10.17) \cite{g4prm} has
 * an extra factor of two in the denominator.
 */
CELER_CONSTEXPR_FUNCTION MevPerLen lpm_constant()
{
    using namespace constants;

    constexpr auto electron_mass_csq = electron_mass * ipow<2>(c_light);

    return native_value_to<MevPerLen>(alpha_fine_structure
                                      * ipow<2>(electron_mass_csq)
                                      / (2 * h_planck * c_light));
}

//---------------------------------------------------------------------------//
// Constant functions for model limits
//---------------------------------------------------------------------------//

//! Maximum energy for EM models to be valid
CELER_CONSTEXPR_FUNCTION units::MevEnergy high_energy_limit()
{
    return units::MevEnergy{1e8};  //! 100 TeV
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
