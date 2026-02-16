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
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Opaque index of a muCF material component
using MuCfMatId = OpaqueId<struct MuCfMaterialComponent_>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
