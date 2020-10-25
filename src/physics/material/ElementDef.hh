//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ElementDef.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/OpaqueId.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Fundamental, invariant properties of an element.
 *
 * Add elemental properties as needed if they apply to more than one physics
 * model. TODO:
 * - atomic shell
 * - isotopic components
 *
 * Note that more than one "element def" can exist for a single atomic number:
 * there might be different enrichments of an element in the problem.
 */
struct ElementDef
{
    int            atomic_number; //!< Z number
    units::AmuMass atomic_mass;   //!< Isotope-weighted average atomic mass

    // COMPUTED PROPERTIES

    real_type cbrt_z;   //!< Z^{1/3}
    real_type cbrt_zzp; //!< (Z (Z + 1))^{1/3}
    real_type log_z;    //!< log Z

    real_type mass_radiation_coeff; //!< 1/X_0 (bremsstrahlung)
};

//---------------------------------------------------------------------------//
} // namespace celeritas
