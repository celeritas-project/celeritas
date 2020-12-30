//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeantPhysicsTableWriter.cc
//---------------------------------------------------------------------------//
#include "GeantPhysicsTableWriter.hh"

#include <fstream>
#include <string>
#include <unordered_map>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>

#include <G4VProcess.hh>
#include <G4VEmProcess.hh>
#include <G4VEnergyLossProcess.hh>
#include <G4VMultipleScattering.hh>
#include <G4SystemOfUnits.hh>
#include <G4PhysicsVectorType.hh>
#include <G4ProcessType.hh>
#include <G4Material.hh>

#include "io/ImportPhysicsTable.hh"
#include "io/ImportPhysicsVector.hh"
#include "base/Assert.hh"
#include "base/Range.hh"
#include "base/Types.hh"
#include "comm/Logger.hh"

using celeritas::ImportModel;
using celeritas::ImportPhysicsVector;
using celeritas::ImportPhysicsVectorType;
using celeritas::ImportProcess;
using celeritas::ImportProcessType;
using celeritas::ImportTableType;
using celeritas::ImportUnits;
using celeritas::PDGNumber;
using celeritas::real_type;

namespace geant_exporter
{
namespace
{
//---------------------------------------------------------------------------//
// ANONYMOUS HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Safely switch from G4PhysicsVectorType to ImportPhysicsVectorType.
 * [See G4PhysicsVectorType.hh]
 */
ImportProcessType to_import_process_type(G4ProcessType g4_process_type)
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
    CHECK_UNREACHABLE;
}

//---------------------------------------------------------------------------//
/*!
 * Safely retrieve the correct process enum from a given string.
 */
ImportProcess to_import_process(const std::string& g4_process_name)
{
    static const std::unordered_map<std::string, ImportProcess> process_map = {
        // clang-format off
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
        // clang-format on
    };
    auto iter = process_map.find(g4_process_name);
    if (iter == process_map.end())
    {
        CELER_LOG(warning) << "Encountered unknown process '"
                           << g4_process_name << "'";
        return ImportProcess::unknown;
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Safely retrieve the correct model enum from a given string.
 */
ImportModel to_import_model(const std::string& g4_model_name)
{
    static const std::unordered_map<std::string, ImportModel> model_map = {
        // clang-format off
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
    if (iter == model_map.end())
    {
        CELER_LOG(warning) << "Encountered unknown model '" << g4_model_name
                           << "'";
        return ImportModel::unknown;
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Safely switch from G4PhysicsVectorType to ImportPhysicsVectorType.
 * [See G4PhysicsVectorType.hh]
 */
ImportPhysicsVectorType
to_import_physics_vector_type(G4PhysicsVectorType g4_vector_type)
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
    CHECK_UNREACHABLE;
}

} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with existing TFile reference
 */
GeantPhysicsTableWriter::GeantPhysicsTableWriter(TFile* root_file)
    : root_file_(root_file)
{
    REQUIRE(root_file);
    this->tree_tables_ = std::make_unique<TTree>("tables", "tables");
    tree_tables_->Branch("ImportPhysicsTable", &(table_));
}

//---------------------------------------------------------------------------//
/*!
 * Write the tables on destruction.
 */
GeantPhysicsTableWriter::~GeantPhysicsTableWriter()
{
    try
    {
        root_file_->Write();
    }
    catch (const std::exception& e)
    {
        CELER_LOG(error) << "Failed to write physics tables: " << e.what();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Add physics tables to the ROOT file from given process and particle.
 */
void GeantPhysicsTableWriter::operator()(const G4ParticleDefinition& particle,
                                         const G4VProcess&           process)
{
    const std::string& process_name = process.GetProcessName();

    // Save process and particle info
    table_.process_type = to_import_process_type(process.GetProcessType());
    table_.process      = to_import_process(process_name);
    table_.particle     = PDGNumber(particle.GetPDGEncoding());

    if (const auto* em_process = dynamic_cast<const G4VEmProcess*>(&process))
    {
        // G4VEmProcess tables
        this->fill_em_tables(*em_process);
    }
    else if (const auto* energy_loss
             = dynamic_cast<const G4VEnergyLossProcess*>(&process))
    {
        // G4VEnergyLossProcess tables
        this->fill_energy_loss_tables(*energy_loss);
    }
    else if (const auto* multiple_scattering
             = dynamic_cast<const G4VMultipleScattering*>(&process))
    {
        // G4VMultipleScattering tables
        this->fill_multiple_scattering_tables(*multiple_scattering);
    }
    else
    {
        CELER_LOG(warning) << "Cannot export process '" << process_name << "'";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write EM process tables to the TTree.
 */
void GeantPhysicsTableWriter::fill_em_tables(const G4VEmProcess& process)
{
    for (auto i : celeritas::range(process.GetNumberOfModels()))
    {
        table_.model = to_import_model(process.GetModelByIndex(i)->GetName());

        // Save potential tables
        this->add_table(process.LambdaTable(),
                        ImportTableType::lambda,
                        ImportUnits::cm_inv);
        this->add_table(process.LambdaTablePrim(),
                        ImportTableType::lambda_prim,
                        ImportUnits::cm_mev_inv);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write energy loss tables to the TTree.
 */
void GeantPhysicsTableWriter::fill_energy_loss_tables(
    const G4VEnergyLossProcess& process)
{
    for (auto i : celeritas::range(process.NumberOfModels()))
    {
        table_.model = to_import_model(process.GetModelByIndex(i)->GetName());

        this->add_table(
            process.DEDXTable(), ImportTableType::dedx, ImportUnits::mev);
        this->add_table(process.DEDXTableForSubsec(),
                        ImportTableType::dedx_subsec,
                        ImportUnits::mev);
        this->add_table(process.DEDXunRestrictedTable(),
                        ImportTableType::dedx_unrestricted,
                        ImportUnits::mev);

        this->add_table(process.IonisationTable(),
                        ImportTableType::ionisation,
                        ImportUnits::mev);
        this->add_table(process.IonisationTableForSubsec(),
                        ImportTableType::ionisation_subsec,
                        ImportUnits::mev);

        this->add_table(process.CSDARangeTable(),
                        ImportTableType::csda_range,
                        ImportUnits::cm);
        this->add_table(process.SecondaryRangeTable(),
                        ImportTableType::secondary_range,
                        ImportUnits::cm);
        this->add_table(process.RangeTableForLoss(),
                        ImportTableType::range,
                        ImportUnits::cm);
        this->add_table(process.InverseRangeTable(),
                        ImportTableType::inverse_range,
                        ImportUnits::cm_inv);

        this->add_table(process.LambdaTable(),
                        ImportTableType::lambda,
                        ImportUnits::cm_inv);
        this->add_table(process.SubLambdaTable(),
                        ImportTableType::sublambda,
                        ImportUnits::cm_inv);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write multiple scattering tables to the TTree.
 */
void GeantPhysicsTableWriter::fill_multiple_scattering_tables(
    const G4VMultipleScattering& process)
{
    // TODO: Figure out a method to get the number of models. Max is 4.
    // Other classes have a NumberOfModels(), but not G4VMultipleScattering
    for (auto i : celeritas::range(4))
    {
        if (G4VEmModel* model = process.GetModelByIndex(i))
        {
            table_.model = to_import_model(model->GetName());
            this->add_table(model->GetCrossSectionTable(),
                            ImportTableType::lambda,
                            ImportUnits::cm_inv);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write data from a geant4 physics table if available.
 *
 * It finishes writing the remaining elements of this->table_ and fills the
 * "tables" TTree.
 */
void GeantPhysicsTableWriter::add_table(const G4PhysicsTable* g4table,
                                        ImportTableType       table_type,
                                        ImportUnits           units)
{
    if (!g4table)
    {
        // Table isn't present
        return;
    }

    table_.table_type = table_type;
    table_.units      = units;

    CELER_LOG(debug) << "Writing " << table_.particle.get() << ": "
                     << to_cstring(table_.process_type) << '.'
                     << to_cstring(table_.process) << '.'
                     << to_cstring(table_.model) << ": "
                     << to_cstring(table_.table_type);

    // Convert units
    constexpr real_type energy_scaling = 1 / MeV;
    real_type           scaling        = 0;
    switch (table_.units)
    {
        case ImportUnits::none:
            scaling = 1;
            break;
        case ImportUnits::cm_inv:
            scaling = cm;
            break;
        case ImportUnits::cm_mev_inv:
            scaling = cm * MeV;
            break;
        case ImportUnits::mev:
            scaling = 1 / MeV;
            break;
        case ImportUnits::cm:
            scaling = 1 / cm;
            break;
    }

    // Save physics vectors
    table_.physics_vectors.clear();
    for (const auto* g4vector : *g4table)
    {
        ImportPhysicsVector import_vec;

        // Populate ImportPhysicsVector and push it back to this->table_
        import_vec.vector_type
            = to_import_physics_vector_type(g4vector->GetType());

        for (auto j : celeritas::range(g4vector->GetVectorLength()))
        {
            import_vec.energy.push_back(g4vector->Energy(j) * energy_scaling);
            import_vec.value.push_back((*g4vector)[j] * scaling);
        }
        table_.physics_vectors.push_back(std::move(import_vec));
    }

    tree_tables_->Fill();
}

//---------------------------------------------------------------------------//
} // namespace geant_exporter
