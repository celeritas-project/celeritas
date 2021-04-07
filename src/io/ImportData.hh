//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "ImportParticle.hh"
#include "ImportElement.hh"
#include "ImportMaterial.hh"
#include "ImportProcess.hh"
#include "GdmlGeometryMap.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Import all the needed data from Geant4.
 * 
 * This struct combines all other import structures into one single object for
 * a simpler read/write interface.
 */
struct ImportData
{
    std::vector<ImportParticle> particles;
    std::vector<ImportElement>  elements;
    std::vector<ImportMaterial> materials;
    std::vector<ImportProcess>  processes;
    GdmlGeometryMap             geometry;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
