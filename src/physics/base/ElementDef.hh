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
#include "base/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Fundamental (static) properties of an element.
 */
struct ElementDef
{
    int            atomic_number;
    units::AmuMass atomic_mass;
    real_type      electron_density; // special units??
    // TODO: span<AtomicShellDef> shells;

    // >>> DERIVATIVE PROPERTIES

    real_type radiation_length_tsai;
    real_type cbrt_z;
    real_type cbrt_zzp;
    real_type log_z;
};

//! Opaque index to ElementDef in the global vector of elements
using ElementDefId = OpaqueId<ElementDef>;

//---------------------------------------------------------------------------//
} // namespace celeritas
