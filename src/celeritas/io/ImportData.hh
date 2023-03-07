//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
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
#include "ImportParticle.hh"
#include "ImportProcess.hh"
#include "ImportSBTable.hh"
#include "ImportVolume.hh"
// IWYU pragma: end_exports

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Common electromagnetic physics parameters (see G4EmParameters.hh).
 *
 * \note Geant4 v11 removed the Spline() option from G4EmParameters.hh.
 */
struct ImportEmParameters
{
    //! Energy loss fluctuation
    bool energy_loss_fluct{false};
    //! LPM effect for bremsstrahlung and pair production
    bool lpm{true};
    //! Integral cross section rejection
    bool integral_approach{true};
    //! Slowing down threshold for linearity assumption
    double linear_loss_limit{0.01};
    //! Whether auger emission should be enabled (valid only for relaxation)
    bool auger{false};

    //! Whether parameters are assigned and valid
    explicit operator bool() const { return linear_loss_limit > 0; }
};

//---------------------------------------------------------------------------//
/*!
 * Particle-dependent parameters related to transportation.
 */
struct ImportTransParameters
{
    //! Number of steps a higher-energy looping track takes before it's killed
    int threshold_trials{10};
    //! Energy below which looping tracks are immediately killed [MeV]
    double important_energy{250};

    //! Whether parameters are assigned and valid
    explicit operator bool() const { return threshold_trials > 0; }
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
 * Seltzer-Berger, Livermore PE, and atomic relaxation data are loaded based on
 * atomic numbers, and thus are stored in maps. To retrieve specific data use
 * \c find(atomic_number) .
 *
 * The parameters related to transportation are particle-dependent and stored
 * in a map where the keys are the PDG number.
 *
 * All units must be converted at import time to be in accordance to the
 * Celeritas' unit standard. Refer to \c base/Units.hh for further information.
 *
 * The "processes" field may be empty for testing applications.
 *
 * \sa celeritas::units
 * \sa ImportParticle
 * \sa ImportElement
 * \sa ImportMaterial
 * \sa ImportProcess
 * \sa ImportVolume
 * \sa RootImporter
 */
struct ImportData
{
    //!@{
    //! \name Type aliases
    using ImportSBMap = std::map<int, ImportSBTable>;
    using ImportLivermorePEMap = std::map<int, ImportLivermorePE>;
    using ImportAtomicRelaxationMap = std::map<int, ImportAtomicRelaxation>;
    using ImportTransParamMap = std::map<int, ImportTransParameters>;
    //!@}

    std::vector<ImportParticle> particles;
    std::vector<ImportElement> elements;
    std::vector<ImportMaterial> materials;
    std::vector<ImportProcess> processes;
    std::vector<ImportMscModel> msc_models;
    std::vector<ImportVolume> volumes;
    ImportEmParameters em_params;
    ImportTransParamMap trans_params;
    ImportSBMap sb_data;
    ImportLivermorePEMap livermore_pe_data;
    ImportAtomicRelaxationMap atomic_relaxation_data;

    explicit operator bool() const
    {
        return !particles.empty() && !elements.empty() && !materials.empty()
               && !volumes.empty();
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
