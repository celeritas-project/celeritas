//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/InputSource.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/ext/GeantImporter.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Import physics and setup from a loaded Geant4 state.
 */
using GeantImport = GeantImportDataSelection;

//---------------------------------------------------------------------------//
/*!
 * Import data from a serialized ROOT/JSON file
 */
struct RootImport
{
    std::string filename;
};

//---------------------------------------------------------------------------//
/*!
 * Combine two input classes by overriding a target input from a source.
 */
struct OverrideImport
{
    Input const* source{nullptr};
    Input* target{nullptr};
};

//---------------------------------------------------------------------------//
/*!
 * Input source.
 */
using InputSource = std::vector < {};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
