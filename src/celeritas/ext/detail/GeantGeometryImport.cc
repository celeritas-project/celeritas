//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantGeometryImport.cc
//---------------------------------------------------------------------------//
#include "GeantGeometryImport.hh"

#include "GeantGeometryImporter.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
void g4_to_vecgeom(G4VPhysicalVolume const* world, bool verbose)
{
    // Convert the geometry to VecGeom
    GeantGeometryImporter converter;
    converter.set_verbose(verbose ? 1 : 0);
    converter.convert_G4_geometry(world);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
