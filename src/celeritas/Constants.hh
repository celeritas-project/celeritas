//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Constants.hh
//! \brief Mathematical, numerical, and physical constants
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Constants.hh"

#include "Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Mathematical, numerical, and physical constants.
 *
 * Some of the physical
 * constants listed here are *exact* numerical values: see \citet{si-2019,
 * https://www.bipm.org/en/publications/si-brochure} for definition of
 * constants and how they relate to the different units.
 *
 * Celeritas            | CLHEP                   | Notes
 * -------------------- | ---------------------   | ------------
 * a0_bohr              | Bohr_radius             | Bohr radius
 * alpha_fine_structure | fine_structure_const    | |
 * atomic_mass          | amu                     | Not the same as 1/avogadro
 * eps_electric         | epsilon0                | Vacuum permittivity
 * h_planck             | h_Planck                | |
 * k_boltzmann          | k_Boltzmann             | |
 * mu_magnetic          | mu0                     | Vacuum permeability
 * na_avogadro          | Avogadro                | [1/mol]
 * r_electron           | classic_electr_radius   | Classical electron radius
 * kcd_luminous         | [none]                  | Lumens per Watt
 * lambdabar_electron   | electron_Compton_length | Reduced Compton wavelength
 * stable_decay_constant| [none]                  | Decay for a stable particle
 *
 * In the CLHEP unit system, the value of the constant \c e_electron is defined
 * to be 1 and \c coulomb is derivative from that. To avoid floating point
 * arithmetic issues that would lead to the "units" and "constants" having
 * different values for it, a special case redefines the value for CLHEP.
 *
 * Some experimental physical constants are derived from the other physical
 * constants, but for consistency and clarity they are presented numerically
 * with the units provided in the CODATA 2018 or 2022 datasets. The
 * \c CELERITAS_CODATA cmake variable determines which of the two datasets is
 * "inlined" into the \c celeritas namespace, allowing fine-grain transition
 * for classes that require it.
 * The \c Constants.test.cc
 * unit tests compare the numerical value against the derivative values inside
 * the celeritas unit system. All experimental values include the final
 * (usually two) imprecise digits; their precision is usually on the order of
 * \f$ 10^{-11} \f$.
 */
namespace constants
{
//---------------------------------------------------------------------------//

#define CELER_ICC inline constexpr Constant

//!@{
//! \name Physical constants with exact value as defined by SI
CELER_ICC c_light = Constant{299792458.} * units::meter / units::second;
CELER_ICC h_planck = Constant{6.62607015e-34} * units::joule * units::second;
#if CELERITAS_UNITS != CELERITAS_UNITS_CLHEP
CELER_ICC e_electron = Constant{1.602176634e-19} * units::coulomb;
#endif
CELER_ICC k_boltzmann = Constant{1.380649e-23} * units::joule / units::kelvin;
CELER_ICC na_avogadro{6.02214076e23};
CELER_ICC kcd_luminous{683};
//!@}

#if CELERITAS_UNITS == CELERITAS_UNITS_CLHEP
//!@{
//! \name Special cases for CLHEP
//! Electron charge is unity by definition
CELER_ICC e_electron{1};
//!@}
#endif

//!@{
//! \name Exact derivative constants
CELER_ICC hbar_planck{h_planck / (2 * pi)};
//!@}

//! Experimental physical constants from CODATA 2006
#if CELERITAS_CODATA == CELERITAS_CODATA_2006
inline
#endif
    namespace codata2006
{
CELER_ICC a0_bohr = Constant{5.2917720859e-11} * units::meter;
CELER_ICC alpha_fine_structure = Constant{7.2973525376e-3};
CELER_ICC atomic_mass = Constant{1.660538782e-24} * units::gram;
CELER_ICC electron_mass = Constant{9.10938215e-28} * units::gram;
CELER_ICC proton_mass = Constant{1.672621637e-24} * units::gram;
CELER_ICC eps_electric = Constant{8.854187817e-12} * units::farad
                         / units::meter;
CELER_ICC mu_magnetic = Constant{1.2566370614e-6} * units::newton
                        / (units::ampere * units::ampere);
CELER_ICC r_electron = Constant{2.8179402894e-15} * units::meter;
CELER_ICC rinf_rydberg = Constant{10973731.568527} / units::meter;
CELER_ICC eh_hartree = Constant{4.35974394e-18} / units::meter;
CELER_ICC lambdabar_electron = Constant{3.8615926459e-13} * units::meter;
}

//! Experimental physical constants from CODATA 2018
#if CELERITAS_CODATA == CELERITAS_CODATA_2018
inline
#endif
    namespace codata2018
{
CELER_ICC a0_bohr = Constant{5.29177210903e-11} * units::meter;
CELER_ICC alpha_fine_structure = Constant{7.2973525693e-3};
CELER_ICC atomic_mass = Constant{1.66053906660e-24} * units::gram;
CELER_ICC electron_mass = Constant{9.1093837015e-28} * units::gram;
CELER_ICC proton_mass = Constant{1.67262192369e-24} * units::gram;
CELER_ICC eps_electric = Constant{8.8541878128e-12} * units::farad
                         / units::meter;
CELER_ICC mu_magnetic = Constant{1.25663706212e-6} * units::newton
                        / (units::ampere * units::ampere);
CELER_ICC r_electron = Constant{2.8179403262e-15} * units::meter;
CELER_ICC rinf_rydberg = Constant{10973731.568160} / units::meter;
CELER_ICC eh_hartree = Constant{4.3597447222071e-18} / units::meter;
CELER_ICC lambdabar_electron = Constant{3.8615926796e-13} * units::meter;
}

//! Experimental physical constants from CODATA 2022
#if CELERITAS_CODATA == CELERITAS_CODATA_2022
inline
#endif
    namespace codata2022
{
CELER_ICC a0_bohr = Constant{5.29177210544e-11} * units::meter;
CELER_ICC alpha_fine_structure = Constant{7.2973525643e-3};
CELER_ICC atomic_mass = Constant{1.66053906892e-24} * units::gram;
CELER_ICC electron_mass = Constant{9.1093837139e-28} * units::gram;
CELER_ICC proton_mass = Constant{1.67262192595e-24} * units::gram;
CELER_ICC eps_electric = Constant{8.8541878188e-12} * units::farad
                         / units::meter;
CELER_ICC mu_magnetic = Constant{1.25663706127e-6} * units::newton
                        / (units::ampere * units::ampere);
CELER_ICC r_electron = Constant{2.8179403205e-15} * units::meter;
CELER_ICC rinf_rydberg = Constant{10973731.568157} / units::meter;
CELER_ICC eh_hartree = Constant{4.3597447222060e-18} / units::meter;
CELER_ICC lambdabar_electron = Constant{3.8615926744e-13} * units::meter;
}

//!@{
//! \name Other constants with physical meaning
inline constexpr int stable_decay_constant{0};
//!@}

#undef CELER_ICC

//---------------------------------------------------------------------------//
}  // namespace constants
}  // namespace celeritas
