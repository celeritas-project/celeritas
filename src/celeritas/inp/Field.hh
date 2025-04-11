//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Field.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>
#include <vector>

#include "geocel/Types.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/field/CylMapFieldInput.hh"
#include "celeritas/field/FieldDriverOptions.hh"
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
 * If volumes are specified, the field will only be present in those volumes.
 *
 * \todo Field driver options will be separate from the magnetic field. They,
 * plus the field type, will be specified in a FieldParams that maps
 * region/particle/energy to field setup. NOTE ALSO that \c
 * driver_options.max_substeps is redundant with \c
 * p.tracking.limits.field_substeps .
 */
struct UniformField
{
    //! Default field units are tesla
    UnitSystem units{UnitSystem::si};

    //! Field strength
    Real3 strength{0, 0, 0};

    //! Field driver options
    FieldDriverOptions driver_options;

    //! Volumes where the field is present (optional)
    std::vector<VolumeId> volumes;
};

//---------------------------------------------------------------------------//
/*!
 * Build a separable R-Z magnetic field from a file.
 *
 * \todo v0.7 Move field input here
 */
using RZMapField = ::celeritas::RZMapFieldInput;
using CylMapField = ::celeritas::CylMapFieldInput;

//---------------------------------------------------------------------------//
//! Field type
using Field = std::variant<NoField, UniformField, RZMapField, CylMapField>;

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
