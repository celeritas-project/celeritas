//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/LoadCovfieField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "celeritas/field/CartMapFieldInput.hh"

namespace celeritas
{
namespace dd
{
//---------------------------------------------------------------------------//
/*!
 * Load a Cartesian magnetic field map from a binary covfie file.
 *
 * The covfie file must have been written with the format:
 *   affine -> nearest_neighbour -> strided -> array (float3)
 *
 * Coordinates in the file are assumed to be in centimetres and field values
 * in tesla; both are converted to Celeritas native units on load.
 *
 * The returned \c CartMapFieldInput has the field driver options left at their
 * defaults (zero). The caller is responsible for setting them from the DD4hep
 * steering-file parameters before passing the input to a factory.
 */
CartMapFieldInput LoadCovfieField(std::string const& filename);

//---------------------------------------------------------------------------//
}  // namespace dd
}  // namespace celeritas
