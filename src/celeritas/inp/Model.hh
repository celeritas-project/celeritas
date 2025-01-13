//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Model.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Set up geometry/material model.
 */
struct Model
{
    //! Path to GDML (or ORANGE override) file, empty to import from Geant4
    std::string geometry_file;

    // TODO: Materials
    // TODO: Regions
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
