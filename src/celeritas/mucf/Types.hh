//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "celeritas/Quantities.hh"

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
using MucfMatId = OpaqueId<struct MucfMaterialComponent_>;

//---------------------------------------------------------------------------//
// REACTIVE SPIN STATES
//---------------------------------------------------------------------------//

//! State F = 1/2 (dd and tt)
inline constexpr units::HalfSpinInt spin_one_half{1};

//! State F = 3/2 (dd)
inline constexpr units::HalfSpinInt spin_three_halves{3};

//! State F = 0 (dt)
inline constexpr units::HalfSpinInt spin_zero{0};

//! State F = 1 (dt)
inline constexpr units::HalfSpinInt spin_one{2};

//---------------------------------------------------------------------------//
}  // namespace celeritas
