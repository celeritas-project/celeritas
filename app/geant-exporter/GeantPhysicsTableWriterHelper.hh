//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeantPhysicsTableWriterHelper.hh
//! Helper functions to safely convert physics tables enums.
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <unordered_map>

#include <G4PhysicsVectorType.hh>
#include <G4ProcessType.hh>
#include <G4Material.hh>

#include "io/ImportTableType.hh"
#include "io/ImportProcessType.hh"
#include "io/ImportProcess.hh"
#include "io/ImportPhysicsVectorType.hh"
#include "io/ImportModel.hh"
#include "io/ImportMaterial.hh"
#include "base/Assert.hh"
#include "base/Types.hh"

using celeritas::ImportMaterialState;
using celeritas::ImportModel;
using celeritas::ImportPhysicsVectorType;
using celeritas::ImportProcess;
using celeritas::ImportProcessType;
using celeritas::ImportTableType;

namespace geant_exporter
{
//---------------------------------------------------------------------------//
/*!
 * Safely retrieve the correct table type enum from a given string.
 */
ImportTableType to_geant_table_type(const std::string& g4_table_type_name)
{
    const static std::unordered_map<std::string, ImportTableType> table_map = {
        // clang-format off
        {"not_defined",  ImportTableType::not_defined},
        {"DEDX",         ImportTableType::dedx},
        {"Ionisation",   ImportTableType::ionisation},
        {"Range",        ImportTableType::range},
        {"RangeSec",     ImportTableType::range_sec},
        {"InverseRange", ImportTableType::inverse_range},
        {"Lambda",       ImportTableType::lambda},
        {"LambdaPrim",   ImportTableType::lambda_prim},
        {"LambdaMod1",   ImportTableType::lambda_mod_1},
        {"LambdaMod2",   ImportTableType::lambda_mod_2},
        {"LambdaMod3",   ImportTableType::lambda_mod_3},
        {"LambdaMod4",   ImportTableType::lambda_mod_4},
        // clang-format on
    };
    auto iter = table_map.find(g4_table_type_name);
    CHECK(iter != table_map.end());
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Safely switch from G4PhysicsVectorType to ImportPhysicsVectorType.
 * [See G4PhysicsVectorType.hh]
 */
ImportPhysicsVectorType
to_geant_physics_vector_type(const G4PhysicsVectorType g4_vector_type)
{
    switch (g4_vector_type)
    {
        case G4PhysicsVectorType::T_G4PhysicsVector:
            return ImportPhysicsVectorType::base;
        case G4PhysicsVectorType::T_G4PhysicsLinearVector:
            return ImportPhysicsVectorType::linear;
        case G4PhysicsVectorType::T_G4PhysicsLogVector:
            return ImportPhysicsVectorType::log;
        case G4PhysicsVectorType::T_G4PhysicsLnVector:
            return ImportPhysicsVectorType::ln;
        case G4PhysicsVectorType::T_G4PhysicsFreeVector:
            return ImportPhysicsVectorType::free;
        case G4PhysicsVectorType::T_G4PhysicsOrderedFreeVector:
            return ImportPhysicsVectorType::ordered_free;
        case G4PhysicsVectorType::T_G4LPhysicsFreeVector:
            return ImportPhysicsVectorType::low_energy_free;
    }
    CHECK(false);
}

//---------------------------------------------------------------------------//
/*!
 * Safely switch from G4PhysicsVectorType to ImportPhysicsVectorType.
 * [See G4PhysicsVectorType.hh]
 */
ImportProcessType to_geant_process_type(const G4ProcessType g4_process_type)
{
    switch (g4_process_type)
    {
        case G4ProcessType::fNotDefined:
            return ImportProcessType::not_defined;
        case G4ProcessType::fTransportation:
            return ImportProcessType::transportation;
        case G4ProcessType::fElectromagnetic:
            return ImportProcessType::electromagnetic;
        case G4ProcessType::fOptical:
            return ImportProcessType::optical;
        case G4ProcessType::fHadronic:
            return ImportProcessType::hadronic;
        case G4ProcessType::fPhotolepton_hadron:
            return ImportProcessType::photolepton_hadron;
        case G4ProcessType::fDecay:
            return ImportProcessType::decay;
        case G4ProcessType::fGeneral:
            return ImportProcessType::general;
        case G4ProcessType::fParameterisation:
            return ImportProcessType::parameterisation;
        case G4ProcessType::fUserDefined:
            return ImportProcessType::user_defined;
        case G4ProcessType::fParallel:
            return ImportProcessType::parallel;
        case G4ProcessType::fPhonon:
            return ImportProcessType::phonon;
        case G4ProcessType::fUCN:
            return ImportProcessType::ucn;
    }
    CHECK(false);
}

//---------------------------------------------------------------------------//
/*!
 * Safely retrieve the correct process enum from a given string.
 */
ImportProcess to_geant_process(const std::string& g4_process_name)
{
    const static std::unordered_map<std::string, ImportProcess> process_map = {
        // clang-format off
        {"not_defined",    ImportProcess::not_defined},
        {"ionIoni",        ImportProcess::ion_ioni},
        {"msc",            ImportProcess::msc},
        {"hIoni",          ImportProcess::h_ioni},
        {"hBrems",         ImportProcess::h_brems},
        {"hPairProd",      ImportProcess::h_pair_prod},
        {"CoulombScat",    ImportProcess::coulomb_scat},
        {"eIoni",          ImportProcess::e_ioni},
        {"eBrem",          ImportProcess::e_brem},
        {"phot",           ImportProcess::photoelectric},
        {"compt",          ImportProcess::compton},
        {"conv",           ImportProcess::conversion},
        {"Rayl",           ImportProcess::rayleigh},
        {"annihil",        ImportProcess::annihilation},
        {"muIoni",         ImportProcess::mu_ioni},
        {"muBrems",        ImportProcess::mu_brems},
        {"muPairProd",     ImportProcess::mu_pair_prod},
        {"Transportation", ImportProcess::transportation},
        // clang-format on
    };
    auto iter = process_map.find(g4_process_name);
    CHECK(iter != process_map.end());
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Safely retrieve the correct model enum from a given string.
 */
ImportModel to_geant_model(const std::string& g4_model_name)
{
    const static std::unordered_map<std::string, ImportModel> model_map = {
        // clang-format off
        {"not_defined",         ImportModel::not_defined},
        {"BraggIon",            ImportModel::bragg_ion},
        {"BetheBloch",          ImportModel::bethe_bloch},
        {"UrbanMsc",            ImportModel::urban_msc},
        {"ICRU73QO",            ImportModel::icru_73_qo},
        {"WentzelVIUni",        ImportModel::wentzel_VI_uni},
        {"hBrem",               ImportModel::h_brem},
        {"hPairProd",           ImportModel::h_pair_prod},
        {"eCoulombScattering",  ImportModel::e_coulomb_scattering},
        {"Bragg",               ImportModel::bragg},
        {"MollerBhabha",        ImportModel::moller_bhabha},
        {"eBremSB",             ImportModel::e_brem_sb},
        {"eBremLPM",            ImportModel::e_brem_lpm},
        {"eplus2gg",            ImportModel::e_plus_to_gg},
        {"LivermorePhElectric", ImportModel::livermore_photoelectric},
        {"Klein-Nishina",       ImportModel::klein_nishina},
        {"BetheHeitlerLPM",     ImportModel::bethe_heitler_lpm},
        {"LivermoreRayleigh",   ImportModel::livermore_rayleigh},
        {"MuBetheBloch",        ImportModel::mu_bethe_bloch},
        {"MuBrem",              ImportModel::mu_brem},
        {"muPairProd",          ImportModel::mu_pair_prod},
        // clang-format on
    };
    auto iter = model_map.find(g4_model_name);
    CHECK(iter != model_map.end());
    return iter->second;
}

//---------------------------------------------------------------------------//
} // namespace geant_exporter
