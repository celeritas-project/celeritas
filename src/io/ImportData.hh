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
#include "ImportVolume.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Import all the needed data from external sources (currently Geant4).
 *
 * All the data imported to Celeritas is stored in this single entity. Any
 * external app should fill this struct and record it in a ROOT file as a
 * single TTree entry, which will be read by \c RootImporter to load the data
 * into Celeritas.
 *
 * \sa ImportParticle
 * \sa ImportElement
 * \sa ImportMaterial
 * \sa ImportProcess
 * \sa RootImporter
 */
struct ImportData
{
    std::vector<ImportParticle> particles;
    std::vector<ImportElement>  elements;
    std::vector<ImportMaterial> materials;
    std::vector<ImportProcess>  processes;
    std::vector<ImportVolume>   volumes;

    explicit operator bool() const
    {
        return !particles.empty() && !elements.empty() && !materials.empty()
               && !processes.empty() && !volumes.empty();
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Return an ImportElement object provided an ImportData and element_id
const ImportElement
get_import_element(const ImportData& data, unsigned int element_id);

// Return an ImportMaterial object provided an ImportData and material_id
const ImportMaterial
get_import_material(const ImportData& data, unsigned int material_id);

//---------------------------------------------------------------------------//
} // namespace celeritas
