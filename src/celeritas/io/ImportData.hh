//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

// IWYU pragma: begin_exports
#include "ImportAtomicRelaxation.hh"
#include "ImportElement.hh"
#include "ImportLivermorePE.hh"
#include "ImportMaterial.hh"
#include "ImportOpticalMaterial.hh"
#include "ImportParameters.hh"
#include "ImportParticle.hh"
#include "ImportProcess.hh"
#include "ImportSBTable.hh"
#include "ImportVolume.hh"
// IWYU pragma: end_exports

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store imported physics data from external sources.
 *
 * All the data imported to Celeritas is stored in this single entity. This
 * struct can be used in memory or recorded in a ROOT TBranch as a single TTree
 * entry, which will be read by \c RootImporter to load the data into
 * Celeritas. Currently, the TTree and TBranch names are hardcoded as \e
 * geant4_data and \e ImportData in \c RootImporter .
 *
 * Each entity's id is defined by its vector position. An \c ImportElement with
 * id = 3 is stored at \c elements[3] . Same for materials and volumes.
 *
 * Seltzer-Berger, Livermore PE, and atomic relaxation data are loaded based on
 * atomic numbers, and thus are stored in maps. To retrieve specific data use
 * \c find(atomic_number) .
 *
 * The unit system of the data is stored in the "units" string. If empty
 * (backward compatibility) or "CGS" the embedded contents are in CGS. If
 * "CLHEP" the units are CLHEP. The \c convert_to_native function will
 * convert a data structure in place and update the units label. Refer to \c
 * base/Units.hh for further information on unit systems.
 *
 * The "processes" field may be empty for testing applications.
 */
struct ImportData
{
    //!@{
    //! \name Type aliases
    using ZInt = int;
    using MatIdx = int;
    using ImportSBMap = std::map<ZInt, ImportSBTable>;
    using ImportLivermorePEMap = std::map<ZInt, ImportLivermorePE>;
    using ImportAtomicRelaxationMap = std::map<ZInt, ImportAtomicRelaxation>;
    using ImportNeutronElasticMap = std::map<ZInt, ImportPhysicsVector>;
    using ImportOpticalMap = std::map<MatIdx, ImportOpticalMaterial>;
    //!@}

    std::vector<ImportParticle> particles;
    std::vector<ImportIsotope> isotopes;
    std::vector<ImportElement> elements;
    std::vector<ImportMaterial> materials;
    std::vector<ImportProcess> processes;
    std::vector<ImportMscModel> msc_models;
    std::vector<ImportVolume> volumes;
    ImportOpticalMap optical;
    ImportEmParameters em_params;
    ImportTransParameters trans_params;
    ImportSBMap sb_data;
    ImportLivermorePEMap livermore_pe_data;
    ImportNeutronElasticMap neutron_elastic_data;
    ImportAtomicRelaxationMap atomic_relaxation_data;

    std::string units;  //!< "cgs", "clhep", or "si"
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Recursively convert imported data to the native unit type
void convert_to_native(ImportData* data);

//---------------------------------------------------------------------------//
}  // namespace celeritas
