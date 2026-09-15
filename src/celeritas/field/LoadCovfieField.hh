//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/LoadCovfieField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "celeritas/inp/Field.hh"

#include "RZMapFieldInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
#if CELERITAS_USE_COVFIE
/*!
 * Load a Cartesian magnetic field map from a binary covfie file.
 *
 * The covfie file must have been written with the stateful pipeline:
 *   affine(3D) -> strided(3D) -> array(float3)
 * with any stateless interpolation backend (e.g. \c linear or \c
 * nearest_neighbour) between the \c affine and \c strided layers.  Stateless
 * backends do not write IO headers, so the binary format is identical
 * regardless of which one was used when writing the file.
 *
 * The returned coordinates and field values are in *native Celeritas units*.
 * No unit conversion is applied; the caller is responsible for ensuring the
 * covfie file was written in the correct unit system or for converting the
 * returned data afterward.
 *
 * The returned \c inp::CartMapField has the field driver options left at their
 * defaults. The caller is responsible for setting them before passing the
 * input to a factory.
 */
inp::CartMapField load_covfie_cart_field(std::string const& filename);

//---------------------------------------------------------------------------//
/*!
 * Load a 2D BrBz cylindrical magnetic field map from a binary covfie file.
 *
 * The covfie file must have been written with the stateful pipeline:
 *   affine(2D) -> strided(2D) -> array(float2)
 * with any stateless interpolation backend between the \c affine and \c
 * strided layers.
 *
 * The two axes are (r, z) and the two field components are (Br, Bz).
 *
 * The returned \c RZMapFieldInput uses [Z][R] indexing (R stride 1),
 * matching the existing Celeritas convention.
 *
 * The returned coordinates and field values are in *native Celeritas units*.
 * No unit conversion is applied.
 *
 * The returned \c RZMapFieldInput has the field driver options left at their
 * defaults. The caller is responsible for setting them before passing the
 * input to a factory.
 */
RZMapFieldInput load_covfie_rz_field(std::string const& filename);

#else

inline inp::CartMapField load_covfie_cart_field(std::string const&)
{
    CELER_NOT_CONFIGURED("covfie");
}

inline RZMapFieldInput load_covfie_rz_field(std::string const&)
{
    CELER_NOT_CONFIGURED("covfie");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
