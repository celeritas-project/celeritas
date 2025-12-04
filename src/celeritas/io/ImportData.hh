//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/inp/Grid.hh"
#include "celeritas/inp/Particle.hh"
#include "celeritas/inp/Physics.hh"
#include "celeritas/inp/PhysicsModel.hh"
// IWYU pragma: begin_exports
#include "ImportElement.hh"
#include "ImportLivermorePE.hh"
#include "ImportMaterial.hh"
#include "ImportOpticalMaterial.hh"
#include "ImportOpticalModel.hh"
#include "ImportParameters.hh"
#include "ImportProcess.hh"
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
 * id = 3 is stored at \c elements[3] . The same is true for
 * geometry/physics/materials (all of which have an independent index!) and
 * volumes.
 *
 * Seltzer-Berger, Livermore PE, and atomic relaxation data are loaded based on
 * atomic numbers, and thus are stored in maps. To retrieve specific data use
 * \c find(atomic_number) .
 *
 * The unit system of the data is stored in the "units" string. If empty
 * (backward compatibility) or "cgs" the embedded contents are in CGS. If
 * "clhep" the units are CLHEP (the native Geant4 unit system). The \c
 * convert_to_native function will convert a data structure in place and update
 * the units label. Refer to \c base/Units.hh for further information on unit
 * systems.
 *
 * \note  \c ImportData will eventually become \c inp::Problem once the import
 * system populates all its \c inp:: objects.
 */
struct ImportData
{
    //!@{
    //! \name Type aliases
    using ZInt = int;
    using GeoMatIndex = unsigned int;
    using ImportSBMap = std::map<ZInt, inp::TwodGrid>;
    //!@}

    //!@{
    //! \name Material data
    std::vector<ImportIsotope> isotopes;
    std::vector<ImportElement> elements;
    std::vector<ImportGeoMaterial> geo_materials;
    std::vector<ImportPhysMaterial> phys_materials;
    //!@}

    //!@{
    //! \name Spatial region data
    std::vector<ImportVolume> volumes;
    //!@}

    //!@{
    //! \name Physics data
    std::vector<inp::Particle> particles;
    std::vector<ImportProcess> processes;
    std::vector<ImportMscModel> msc_models;
    //!@}

    //!@{
    //! \name Physics configuration options
    ImportEmParameters em_params;
    ImportTransParameters trans_params;
    //!@}

    //!@{
    //! \name Optical data
    ImportOpticalParameters optical_params;
    std::vector<ImportOpticalModel> optical_models;
    std::vector<ImportOpticalMaterial> optical_materials;
    //!@}

    //! Unit system of the stored data: "cgs", "clhep", or "si"
    std::string units;

    //!@{
    //! \name Current \c inp::Problem data that has been ported

    // Models
    inp::LivermorePhotoModel livermore_photo;
    inp::MuPairProductionModel mu_production;
    inp::SeltzerBergerModel seltzer_berger;

    // Processes
    inp::AtomicRelaxation atomic_relaxation;

    // Physics groups
    inp::OpticalPhysics optical_physics;

    // Muon-catalyzed fusion
    inp::MucfPhysics mucf_physics;
    //!@}
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Recursively convert imported data to the native unit type
void convert_to_native(ImportData* data);

// Whether an imported model of the given class is present
bool has_model(ImportData const&, ImportModelClass);

// Whether an imported MSC model of the given class is present
bool has_msc_model(ImportData const&, ImportModelClass);

//---------------------------------------------------------------------------//
}  // namespace celeritas
