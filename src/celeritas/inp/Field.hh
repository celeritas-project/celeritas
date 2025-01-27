//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Field.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>

#include "geocel/Types.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/field/RZMapFieldInput.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Build a problem without magnetic fields.
 */
struct NoField
{
};

//---------------------------------------------------------------------------//
/*!
 * Create a uniform nonzero field.
 *
 * \todo Field driver options will be separate from the magnetic field. They,
 * plus the field type, will be specified in a FieldParams that maps
 * region/particle/energy to field setup.
 */
struct UniformField
{
    //! Default field units are tesla
    UnitSystem units{UnitSystem::si};

    //! Field strength
    Real3 strength{0, 0, 0};

    //! Field driver options
    FieldDriverOptions driver_options;
};

//---------------------------------------------------------------------------//
/*!
 * Build a separable R-Z magnetic field from a file.
 *
 * \todo Move field input here
 */
using RZMapField = ::celeritas::RZMapFieldInput;

//---------------------------------------------------------------------------//
//! Field type
using Field = std::variant<NoField, UniformField, RZMapField>;

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
