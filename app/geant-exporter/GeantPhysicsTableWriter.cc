//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeantPhysicsTableWriter.cc
//---------------------------------------------------------------------------//
#include "GeantPhysicsTableWriter.hh"

#include <fstream>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>

#include <G4VProcess.hh>
#include <G4VEmProcess.hh>
#include <G4VEnergyLossProcess.hh>
#include <G4VMultipleScattering.hh>
#include <G4SystemOfUnits.hh>

#include "GeantPhysicsTableWriterHelper.hh"
#include "io/ImportPhysicsTable.hh"
#include "io/ImportPhysicsVector.hh"
#include "io/ImportTableType.hh"
#include "io/ImportProcessType.hh"
#include "io/ImportProcess.hh"
#include "io/ImportPhysicsVectorType.hh"
#include "io/ImportModel.hh"
#include "base/Range.hh"
#include "base/Types.hh"

using celeritas::ImportModel;
using celeritas::ImportPhysicsVector;
using celeritas::ImportPhysicsVectorType;
using celeritas::ImportProcess;
using celeritas::ImportProcessType;
using celeritas::ImportTableType;
using celeritas::PDGNumber;
using celeritas::real_type;
using geant_exporter::to_geant_model;
using geant_exporter::to_geant_physics_vector_type;
using geant_exporter::to_geant_process;
using geant_exporter::to_geant_process_type;
using geant_exporter::to_geant_table_type;

namespace geant_exporter
{
//---------------------------------------------------------------------------//
/*!
 * Construct with existing TFile reference
 */
GeantPhysicsTableWriter::GeantPhysicsTableWriter(TFile* root_file)
{
    REQUIRE(root_file);
    this->tree_tables_ = std::make_unique<TTree>("tables", "tables");
    tree_tables_->Branch("ImportPhysicsTable", &(table_));
}

//---------------------------------------------------------------------------//
/*!
 * Add physics tables to the ROOT file from given process and particle
 */
void GeantPhysicsTableWriter::add_physics_tables(
    const G4VProcess& process, const G4ParticleDefinition& particle)
{
    // Process name
    const std::string& process_name = process.GetProcessName();

    // Write this->table_
    table_.process_type = to_geant_process_type(process.GetProcessType());
    table_.process      = to_geant_process(process_name);
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
        std::cout << "  No available code for " << process_name << std::endl;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write EM process tables to the TTree
 */
void GeantPhysicsTableWriter::fill_em_tables(const G4VEmProcess& em_process)
{
    const std::string process_name = em_process.GetProcessName();

    for (auto i : celeritas::range(em_process.GetNumberOfModels()))
    {
        // Model
        const std::string& model_name
            = em_process.GetModelByIndex(i)->GetName();

        // Write table_
        table_.model = to_geant_model(model_name);

        std::string table_name = process_name + "_" + model_name;

        // The same model can have both Lambda and LambdaPrim tables
        if (const G4PhysicsTable* table = em_process.LambdaTable())
        {
            this->fill_tables_tree(
                *table, "Lambda", table_name, ImportPhysicsVector::DataType::xs);
        }

        if (const G4PhysicsTable* table = em_process.LambdaTablePrim())
        {
            this->fill_tables_tree(*table,
                                   "LambdaPrim",
                                   table_name,
                                   ImportPhysicsVector::DataType::xs);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write energy loss tables to the TTree
 */
void GeantPhysicsTableWriter::fill_energy_loss_tables(
    const G4VEnergyLossProcess& eloss_process)
{
    const std::string& process_name = eloss_process.GetProcessName();

    for (auto i : celeritas::range(eloss_process.NumberOfModels()))
    {
        // Model
        const std::string& model_name
            = eloss_process.GetModelByIndex(i)->GetName();

        // Write table_
        table_.model = to_geant_model(model_name);

        std::string table_name = process_name + "_" + model_name;

        if (const G4PhysicsTable* table = eloss_process.DEDXTable())
        {
            this->fill_tables_tree(*table,
                                   "DEDX",
                                   table_name,
                                   ImportPhysicsVector::DataType::energy_loss);
        }

        if (const G4PhysicsTable* table = eloss_process.DEDXTableForSubsec())
        {
            this->fill_tables_tree(*table,
                                   "SubDEDX",
                                   table_name,
                                   ImportPhysicsVector::DataType::energy_loss);
        }

        if (const G4PhysicsTable* table = eloss_process.DEDXunRestrictedTable())
        {
            this->fill_tables_tree(*table,
                                   "DEDXnr",
                                   table_name,
                                   ImportPhysicsVector::DataType::energy_loss);
        }

        if (const G4PhysicsTable* table = eloss_process.IonisationTable())
        {
            this->fill_tables_tree(*table,
                                   "Ionisation",
                                   table_name,
                                   ImportPhysicsVector::DataType::energy_loss);
        }

        if (const G4PhysicsTable* table
            = eloss_process.IonisationTableForSubsec())
        {
            this->fill_tables_tree(*table,
                                   "SubIonisation",
                                   table_name,
                                   ImportPhysicsVector::DataType::energy_loss);
        }

        if (const G4PhysicsTable* table = eloss_process.CSDARangeTable())
        {
            this->fill_tables_tree(*table,
                                   "CSDARange",
                                   table_name,
                                   ImportPhysicsVector::DataType::energy_loss);
        }

        if (const G4PhysicsTable* table = eloss_process.SecondaryRangeTable())
        {
            this->fill_tables_tree(*table,
                                   "RangeSec",
                                   table_name,
                                   ImportPhysicsVector::DataType::energy_loss);
        }

        if (G4PhysicsTable* table = eloss_process.RangeTableForLoss())
        {
            this->fill_tables_tree(*table,
                                   "Range",
                                   table_name,
                                   ImportPhysicsVector::DataType::energy_loss);
        }

        if (const G4PhysicsTable* table = eloss_process.InverseRangeTable())
        {
            this->fill_tables_tree(*table,
                                   "InverseRange",
                                   table_name,
                                   ImportPhysicsVector::DataType::energy_loss);
        }

        if (const G4PhysicsTable* table = eloss_process.LambdaTable())
        {
            this->fill_tables_tree(
                *table, "Lambda", table_name, ImportPhysicsVector::DataType::xs);
        }

        if (const G4PhysicsTable* table = eloss_process.SubLambdaTable())
        {
            this->fill_tables_tree(*table,
                                   "SubLambda",
                                   table_name,
                                   ImportPhysicsVector::DataType::xs);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write multiple scattering tables to the TTree
 */
void GeantPhysicsTableWriter::fill_multiple_scattering_tables(
    const G4VMultipleScattering& msc_process)
{
    const std::string& process_name = msc_process.GetProcessName();

    // TODO: Figure out a method to get the number of models. Max is 4.
    // Other classes have a NumberOfModels(), but not G4VMultipleScattering
    for (auto i : celeritas::range(4))
    {
        if (G4VEmModel* model = msc_process.GetModelByIndex(i))
        {
            if (const G4PhysicsTable* table = model->GetCrossSectionTable())
            {
                // Table type
                const std::string& model_name = model->GetName();

                // Write table_
                table_.model = to_geant_model(model_name);

                std::string table_name      = process_name + "_" + model_name;
                std::string table_type_name = "LambdaMod"
                                              + std::to_string(i + 1);
                this->fill_tables_tree(
                    *table,
                    table_type_name,
                    table_name,
                    ImportPhysicsVector::DataType::energy_loss);
            }
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write a given G4PhysicsTable as a ImportPhysicsVector in this->table_
 */
void GeantPhysicsTableWriter::fill_physics_vectors(
    const G4PhysicsTable& table, ImportPhysicsVector::DataType xs_or_eloss)
{
    // Clean up so previous vector data is not carried forward
    table_.physics_vectors.clear();

    // Loop over G4PhysicsTable
    for (const auto* phys_vector : table)
    {
        ImportPhysicsVector geant_physics_vector;

        // Populate ImportPhysicsVector and push it back to this->table_
        geant_physics_vector.vector_type
            = to_geant_physics_vector_type(phys_vector->GetType());

        geant_physics_vector.data_type = xs_or_eloss;

        for (auto j : celeritas::range(phys_vector->GetVectorLength()))
        {
            // Code interface for changing G4PhysicsVector data units
            real_type energy   = phys_vector->Energy(j) / MeV;
            real_type xs_eloss = (*phys_vector)[j];

            if (geant_physics_vector.data_type
                == ImportPhysicsVector::DataType::energy_loss)
            {
                xs_eloss /= MeV;
            }
            else
            {
                xs_eloss /= (1 / cm);
            }

            geant_physics_vector.energy.push_back(energy);     // [MeV]
            geant_physics_vector.xs_eloss.push_back(xs_eloss); // [1/cm or MeV]
        }
        table_.physics_vectors.push_back(geant_physics_vector);
    }
}

//---------------------------------------------------------------------------//
/*!
 * To be called after a G4PhysicsTable has been assigned.
 * It finishes writing the remaining elements of this->table_ and fills the
 * "tables" TTree.
 */
void GeantPhysicsTableWriter::fill_tables_tree(
    const G4PhysicsTable&         table,
    std::string                   table_type_name,
    std::string                   table_name,
    ImportPhysicsVector::DataType xs_or_eloss)
{
    // Convert table type
    table_.table_type = to_geant_table_type(table_type_name);

    // Populate this->table_.physics_vectors and fill the TTree
    fill_physics_vectors(table, xs_or_eloss);
    tree_tables_->Fill();

    // Print message
    table_name = table_type_name + "_" + table_name;
    std::cout << "  Added " << table_name << std::endl;
}

//---------------------------------------------------------------------------//
} // namespace geant_exporter
