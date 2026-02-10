//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//
/*!
 * Muonic atom selection from material data. This is *not* intended to be used
 * by the transport loop.
 */
enum class MucfIsotope
{
    protium,
    deuterium,
    tritium,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Muonic atom selection from material data. This is *not* intended to be used
 * by the transport loop.
 */
enum class MucfMuonicAtom
{
    deuterium,
    tritium,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Muonic molecule selection from material data. This is *not* intended to be
 * used by the transport loop.
 */
enum class MucfMuonicMolecule
{
    deuterium_deuterium,
    deuterium_tritium,
    tritium_tritium,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Enum for safely accessing hydrogen isoprotologues.
 *
 * Hydrogen isoprotologue molecules are:
 * - Homonuclear: \f$ ^2H \f$, \f$ ^2d \f$, and \f$ ^2t \f$
 * - Heteronuclear: hd, ht, and dt.
 *
 * \note Muon-catalyzed fusion data is only applicable to a material with
 * concentrations in thermodynamic equilibrium. This equilibrium is calculated
 * at model construction from the material temperature and its h, d, and t
 * fractions.
 */
enum class MucfIsoprotologueMolecule
{
    protium_protium,
    protium_deuterium,
    protium_tritium,
    deuterium_tritium,
    tritium_tritium,
    size_
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Opaque index of a muCF material component
using MuCfMatId = OpaqueId<struct MuCfMaterialComponent_>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
