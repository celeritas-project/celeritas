//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/OpaqueId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Physical state of matter.
 */
enum class MatterState
{
    unspecified = 0,
    solid,
    liquid,
    gas
};

//---------------------------------------------------------------------------//

//! Opaque index to ElementDef in the global vector of elements
using ElementDefId = OpaqueId<struct ElementDef>;

//! Opaque index to one elemental component datum in a particular material
using ElementComponentId = OpaqueId<struct MatElementComponent>;

//! Opaque index to MaterialDef in a vector: represents a material ID
using MaterialDefId = OpaqueId<struct MaterialDef>;

//---------------------------------------------------------------------------//
} // namespace celeritas
