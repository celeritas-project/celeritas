//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "ImportElement.hh"
#include "ImportLivermorePE.hh"
#include "ImportMaterial.hh"
#include "ImportParticle.hh"
#include "ImportProcess.hh"
#include "ImportSBTable.hh"
#include "ImportVolume.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Enumerator for EM parameters (see G4EmParameters.hh).
 *
 * \note
 * Geant4 v11 removed the Spline() option from G4EmParameters.hh.
 */
enum class ImportEmParameter
{
    energy_loss_fluct, //!< Energy loss fluctuation flag
    lpm,               //!< LPM effect flag (bremsstrahlung, pair production)
    integral_approach, //!< Integral approach flag
    linear_loss_limit, //!< Linear loss limit
    bins_per_decade,   //!< Cross-section table binning
    min_table_energy,  //!< Cross-section table minimum kinetic energy [MeV]
    max_table_energy,  //!< Cross-section table maximum kinetic energy [MeV]
};

//---------------------------------------------------------------------------//
/*!
 * Import all the needed data from external sources (currently Geant4).
 *
 * All the data imported to Celeritas is stored in this single entity. This
 * struct can be used in memory or recorded in a ROOT TBranch as a single TTree
 * entry, which will be read by \c RootImporter to load the data into
 * Celeritas. Currently, the TTree and TBranch names are hardcoded as \e
 * geant4_data and \e ImportData in \c RootImporter .
 *
 * Each entity's id is defined by its vector position. An \c ImportElement with
 * id = 3 is stored at \c elements.at(3) . Same for materials and volumes.
 *
 * All units must be converted at import time to be in accordance to the
 * Celeritas' unit standard. Refer to \c base/Units.hh for further information.
 *
 * The "processes" field may be empty for testing applications.
 *
 * \sa base/Units
 * \sa ImportParticle
 * \sa ImportElement
 * \sa ImportMaterial
 * \sa ImportProcess
 * \sa ImportVolume
 * \sa RootImporter
 * \sa app/celer-export-geant
 */
struct ImportData
{
    //!@{
    //! Type aliases
    using AtomicNumber         = int;
    using ImportEmParamsMap    = std::map<ImportEmParameter, double>;
    using ImportSBMap          = std::map<AtomicNumber, ImportSBTable>;
    using ImportLivermorePEMap = std::map<AtomicNumber, ImportLivermorePE>;
    //!@}

    std::vector<ImportParticle> particles;
    std::vector<ImportElement>  elements;
    std::vector<ImportMaterial> materials;
    std::vector<ImportProcess>  processes;
    std::vector<ImportVolume>   volumes;
    ImportEmParamsMap           em_params;
    ImportSBMap                 sb_data;
    ImportLivermorePEMap        livermore_pe_data;

    explicit operator bool() const
    {
        return !particles.empty() && !elements.empty() && !materials.empty()
               && !volumes.empty() && !em_params.empty()
               && sb_data.size() == elements.size()
               && livermore_pe_data.size() == elements.size();
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

const char* to_cstring(ImportEmParameter value);

//---------------------------------------------------------------------------//
} // namespace celeritas
