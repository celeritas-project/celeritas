//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "detail/ImportParticle.hh"
#include "detail/ImportElement.hh"
#include "detail/ImportMaterial.hh"
#include "detail/ImportProcess.hh"
#include "detail/GdmlGeometryMap.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Import all the needed data from external sources (currently Geant4).
 *
 * This struct combines all import structs and classes into one single object
 * for a simpler read/write interface with ROOT and Celeritas.
 *
 * All the data imported to Celeritas is stored in this single struct. Any
 * external app should fill this struct and record it in a ROOT TBranch, which
 * will be read by \c RootImporter to load the data into Celeritas.
 *
 * \sa ImportParticle
 * \sa ImportElement
 * \sa ImportMaterial
 * \sa ImportProcess
 * \sa GdmlGeometryMap
 */
struct ImportData
{
    std::vector<ImportParticle> particles;
    std::vector<ImportElement>  elements;
    std::vector<ImportMaterial> materials;
    std::vector<ImportProcess>  processes;
    GdmlGeometryMap             geometry;

    explicit operator bool() const
    {
        return !particles.empty() && !elements.empty() && !materials.empty()
               && !processes.empty() && geometry;
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
