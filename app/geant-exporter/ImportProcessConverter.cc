//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportProcessConverter.cc
//---------------------------------------------------------------------------//
#include "ImportProcessConverter.hh"

#include <fstream>
#include <string>

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

#include "io/ImportProcess.hh"
#include "io/ImportPhysicsTable.hh"
#include "io/ImportPhysicsVector.hh"
#include "base/Assert.hh"
#include "base/Range.hh"
#include "base/TypeDemangler.hh"
#include "base/Types.hh"
#include "comm/Logger.hh"
#include "CeleritasG4Version.hh"

using celeritas::ImportModelClass;
using celeritas::ImportPhysicsTable;
using celeritas::ImportPhysicsVector;
using celeritas::ImportPhysicsVectorType;
using celeritas::ImportProcessClass;
using celeritas::ImportProcessType;
using celeritas::ImportTableType;
using celeritas::ImportUnits;
using celeritas::PDGNumber;
using ProcessTypeDemangler = celeritas::TypeDemangler<G4VProcess>;

namespace geant_exporter
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Safely switch from \c G4PhysicsVectorType to \c ImportPhysicsVectorType .
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
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Safely retrieve the correct process enum from a given string.
 */
ImportProcessClass to_import_process_class(const G4VProcess& process)
{
    static const std::unordered_map<std::string, ImportProcessClass> process_map
        = {
            // clang-format off
            {"ionIoni",        ImportProcessClass::ion_ioni},
            {"msc",            ImportProcessClass::msc},
            {"hIoni",          ImportProcessClass::h_ioni},
            {"hBrems",         ImportProcessClass::h_brems},
            {"hPairProd",      ImportProcessClass::h_pair_prod},
            {"CoulombScat",    ImportProcessClass::coulomb_scat},
            {"eIoni",          ImportProcessClass::e_ioni},
            {"eBrem",          ImportProcessClass::e_brems},
            {"phot",           ImportProcessClass::photoelectric},
            {"compt",          ImportProcessClass::compton},
            {"conv",           ImportProcessClass::conversion},
            {"Rayl",           ImportProcessClass::rayleigh},
            {"annihil",        ImportProcessClass::annihilation},
            {"muIoni",         ImportProcessClass::mu_ioni},
            {"muBrems",        ImportProcessClass::mu_brems},
            {"muPairProd",     ImportProcessClass::mu_pair_prod},
            // clang-format on
        };
    auto iter = process_map.find(process.GetProcessName());
    if (iter == process_map.end())
    {
        CELER_LOG(warning) << "Encountered unknown process '"
                           << process.GetProcessName() << "'";
        return ImportProcessClass::unknown;
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Safely retrieve the correct model enum from a given string.
 */
ImportModelClass to_import_model(const std::string& g4_model_name)
{
    static const std::unordered_map<std::string, ImportModelClass> model_map = {
        // clang-format off
        {"BraggIon",            ImportModelClass::bragg_ion},
        {"BetheBloch",          ImportModelClass::bethe_bloch},
        {"UrbanMsc",            ImportModelClass::urban_msc},
        {"ICRU73QO",            ImportModelClass::icru_73_qo},
        {"WentzelVIUni",        ImportModelClass::wentzel_VI_uni},
        {"hBrem",               ImportModelClass::h_brems},
        {"hPairProd",           ImportModelClass::h_pair_prod},
        {"eCoulombScattering",  ImportModelClass::e_coulomb_scattering},
        {"Bragg",               ImportModelClass::bragg},
        {"MollerBhabha",        ImportModelClass::moller_bhabha},
        {"eBremSB",             ImportModelClass::e_brems_sb},
        {"eBremLPM",            ImportModelClass::e_brems_lpm},
        {"eplus2gg",            ImportModelClass::e_plus_to_gg},
        {"LivermorePhElectric", ImportModelClass::livermore_photoelectric},
        {"KleinNishina",        ImportModelClass::klein_nishina},
        {"BetheHeitler",        ImportModelClass::bethe_heitler},
        {"BetheHeitlerLPM",     ImportModelClass::bethe_heitler_lpm},
        {"LivermoreRayleigh",   ImportModelClass::livermore_rayleigh},
        {"MuBetheBloch",        ImportModelClass::mu_bethe_bloch},
        {"MuBrem",              ImportModelClass::mu_brems},
        {"muPairProd",          ImportModelClass::mu_pair_prod},
        // clang-format on
    };
    auto iter = model_map.find(g4_model_name);
    if (iter == model_map.end())
    {
        CELER_LOG(warning) << "Encountered unknown model '" << g4_model_name
                           << "'";
        return ImportModelClass::unknown;
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Safely switch from \c G4PhysicsVectorType to \c ImportPhysicsVectorType .
 * [See G4PhysicsVectorType.hh]
 *
 * Geant4 v11 has a different set of G4PhysicsVectorType enums.
 */
ImportPhysicsVectorType
to_import_physics_vector_type(G4PhysicsVectorType g4_vector_type)
{
    // Geant4 v10
    switch (g4_vector_type)
    {
        case T_G4PhysicsVector:
            return ImportPhysicsVectorType::unknown;
        case T_G4PhysicsLinearVector:
            return ImportPhysicsVectorType::linear;
        case T_G4PhysicsLogVector:
#if CELERITAS_G4_V10
        case T_G4PhysicsLnVector:
#endif
            return ImportPhysicsVectorType::log;
        case T_G4PhysicsFreeVector:
#if CELERITAS_G4_V10
        case T_G4PhysicsOrderedFreeVector:
        case T_G4LPhysicsFreeVector:
#endif
            return ImportPhysicsVectorType::free;
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Get a multiplicative geant-natural-units constant to convert the units.
 */
double units_to_scaling(ImportUnits units)
{
    switch (units)
    {
        case ImportUnits::none:
            return 1;
        case ImportUnits::cm_inv:
            return cm;
        case ImportUnits::cm_mev_inv:
            return cm * MeV;
        case ImportUnits::mev:
            return 1 / MeV;
        case ImportUnits::mev_per_cm:
            return cm / MeV;
        case ImportUnits::cm:
            return 1 / cm;
    }
    CELER_ASSERT_UNREACHABLE();
}

} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with a selected list of tables.
 */
ImportProcessConverter::ImportProcessConverter(TableSelection which_tables)
    : which_tables_(which_tables)
{
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
ImportProcessConverter::~ImportProcessConverter() = default;

//---------------------------------------------------------------------------//
/*!
 * Add physics tables to this->process_ from a given particle and process and
 * return it. If the process was already returned, \c operator() will return an
 * empty object.
 */
ImportProcess
ImportProcessConverter::operator()(const G4ParticleDefinition& particle,
                                   const G4VProcess&           process)
{
    // Check for duplicate processes
    auto iter_ok = written_processes_.insert({&process, {&particle}});
    if (!iter_ok.second)
    {
        const PrevProcess& prev = iter_ok.first->second;
        CELER_LOG(warning) << "Skipping process '" << process.GetProcessName()
                           << "' (" << ProcessTypeDemangler()(process)
                           << ") for particle " << particle.GetParticleName()
                           << ": duplicate of particle "
                           << prev.particle->GetParticleName();
        return {};
    }
    CELER_LOG(debug) << "Saving process '" << process.GetProcessName()
                     << "' for particle " << particle.GetParticleName() << " ("
                     << particle.GetPDGEncoding() << ')';

    // Save process and particle info
    process_.process_type  = to_import_process_type(process.GetProcessType());
    process_.process_class = to_import_process_class(process);
    process_.particle_pdg  = particle.GetPDGEncoding();
    process_.models.clear();
    process_.tables.clear();

    if (const auto* em_process = dynamic_cast<const G4VEmProcess*>(&process))
    {
        // G4VEmProcess tables
        this->store_em_tables(*em_process);
    }
    else if (const auto* energy_loss
             = dynamic_cast<const G4VEnergyLossProcess*>(&process))
    {
        // G4VEnergyLossProcess tables
        this->store_energy_loss_tables(*energy_loss);
    }
    else if (const auto* multiple_scattering
             = dynamic_cast<const G4VMultipleScattering*>(&process))
    {
        // G4VMultipleScattering tables
        this->store_multiple_scattering_tables(*multiple_scattering);
    }
    else
    {
        CELER_LOG(error) << "Cannot export unknown process '"
                         << process.GetProcessName() << "' ("
                         << ProcessTypeDemangler()(process) << ")";
    }

    return process_;
}

//---------------------------------------------------------------------------//
/*!
 * Store EM XS tables to this->process_.
 */
void ImportProcessConverter::store_em_tables(const G4VEmProcess& process)
{
#if CELERITAS_G4_V10
    for (auto i : celeritas::range(process.GetNumberOfModels()))
#else
    for (auto i : celeritas::range(process.NumberOfModels()))
#endif
    {
        process_.models.push_back(
            to_import_model(process.GetModelByIndex(i)->GetName()));
    }

    // Save potential tables
    this->add_table(process.LambdaTable(), ImportTableType::lambda);
    this->add_table(process.LambdaTablePrim(), ImportTableType::lambda_prim);
}

//---------------------------------------------------------------------------//
/*!
 * Store energy loss XS tables to this->process_.
 *
 * The following XS tables do not exist in Geant4 v11.
 * - DEDXTableForSubsec()
 * - IonisationTableForSubsec()
 * - SubLambdaTable()
 */
void ImportProcessConverter::store_energy_loss_tables(
    const G4VEnergyLossProcess& process)
{
    for (auto i : celeritas::range(process.NumberOfModels()))
    {
        process_.models.push_back(
            to_import_model(process.GetModelByIndex(i)->GetName()));
    }

    if (process.IsIonisationProcess())
    {
        // The de/dx and range tables created by summing the contribution from
        // each energy loss process are stored in the "ionization process"
        // (which might be ionization or might be another arbitrary energy loss
        // process if there is no ionization in the problem).
        this->add_table(process.DEDXTable(), ImportTableType::dedx);
        this->add_table(process.RangeTableForLoss(), ImportTableType::range);
    }
    this->add_table(process.LambdaTable(), ImportTableType::lambda);

    if (which_tables_ > TableSelection::minimal)
    {
        // Inverse range is redundant with range
        this->add_table(process.InverseRangeTable(),
                        ImportTableType::inverse_range);

        // None of these tables appear to be used in Geant4
        if (process.IsIonisationProcess())
        {
            // The "ionization table" is just the per-process de/dx table for
            // ionization
            this->add_table(process.IonisationTable(),
                            ImportTableType::dedx_process);
        }
        else
        {
            this->add_table(process.DEDXTable(), ImportTableType::dedx_process);
        }
#if CELERITAS_G4_V10
        this->add_table(process.DEDXTableForSubsec(),
                        ImportTableType::dedx_subsec);
        this->add_table(process.IonisationTableForSubsec(),
                        ImportTableType::ionization_subsec);
        this->add_table(process.SubLambdaTable(), ImportTableType::sublambda);
#endif
        this->add_table(process.DEDXunRestrictedTable(),
                        ImportTableType::dedx_unrestricted);
        this->add_table(process.CSDARangeTable(), ImportTableType::csda_range);
        this->add_table(process.SecondaryRangeTable(),
                        ImportTableType::secondary_range);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Store multiple scattering XS tables to this->process_.
 *
 * Whereas other EM processes combine the model tables into a single process
 * table, MSC keeps them independent.
 *
 * Starting on Geant4 v11, G4MultipleScattering provides \c NumberOfModels() .
 */
void ImportProcessConverter::store_multiple_scattering_tables(
    const G4VMultipleScattering& process)
{
#if CELERITAS_G4_V10
    for (auto i : celeritas::range(4))
#else
    for (int i : celeritas::range(process.NumberOfModels()))
#endif
    {
        if (G4VEmModel* model = process.GetModelByIndex(i))
        {
            process_.models.push_back(to_import_model(model->GetName()));
            this->add_table(model->GetCrossSectionTable(),
                            ImportTableType::lambda);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write data from a Geant4 physics table if available.
 */
void ImportProcessConverter::add_table(const G4PhysicsTable* g4table,
                                       ImportTableType       table_type)
{
    if (!g4table)
    {
        // Table isn't present
        return;
    }

    // Check for duplicate tables
    auto iter_ok = written_tables_.insert(
        {g4table, {process_.particle_pdg, process_.process_class, table_type}});
    if (!iter_ok.second)
    {
        const PrevTable& prev = iter_ok.first->second;
        CELER_LOG(warning) << "Skipping table " << process_.particle_pdg << '.'
                           << to_cstring(process_.process_class) << '.'
                           << to_cstring(table_type) << ": duplicate of "
                           << prev.particle_pdg << '.'
                           << to_cstring(prev.process_class) << '.'
                           << to_cstring(prev.table_type);
        return;
    }

    CELER_LOG(debug) << "Saving table " << process_.particle_pdg << '.'
                     << to_cstring(process_.process_type) << '.'
                     << to_cstring(process_.process_class) << '.'
                     << to_cstring(table_type);

    ImportPhysicsTable table;
    table.table_type = table_type;
    switch (table_type)
    {
        case ImportTableType::dedx:
        case ImportTableType::dedx_process:
        case ImportTableType::dedx_subsec:
        case ImportTableType::dedx_unrestricted:
        case ImportTableType::ionization:
        case ImportTableType::ionization_subsec:
            table.x_units = ImportUnits::mev;
            table.y_units = ImportUnits::mev_per_cm;
            break;
        case ImportTableType::csda_range:
        case ImportTableType::range:
        case ImportTableType::secondary_range:
            table.x_units = ImportUnits::mev;
            table.y_units = ImportUnits::cm;
            break;
        case ImportTableType::inverse_range:
            table.x_units = ImportUnits::cm;
            table.y_units = ImportUnits::mev;
            break;
        case ImportTableType::lambda:
        case ImportTableType::sublambda:
            table.x_units = ImportUnits::mev;
            table.y_units = ImportUnits::cm_inv;
            break;
        case ImportTableType::lambda_prim:
            table.x_units = ImportUnits::mev;
            table.y_units = ImportUnits::cm_mev_inv;
            break;
    };

    // Convert units
    double x_scaling = units_to_scaling(table.x_units);
    double y_scaling = units_to_scaling(table.y_units);

    // Save physics vectors
    for (const auto* g4vector : *g4table)
    {
        ImportPhysicsVector import_vec;

        // Populate ImportPhysicsVectors
        import_vec.vector_type
            = to_import_physics_vector_type(g4vector->GetType());

        for (auto j : celeritas::range(g4vector->GetVectorLength()))
        {
            import_vec.x.push_back(g4vector->Energy(j) * x_scaling);
            import_vec.y.push_back((*g4vector)[j] * y_scaling);
        }
        table.physics_vectors.push_back(std::move(import_vec));
    }

    process_.tables.push_back(std::move(table));
}

//---------------------------------------------------------------------------//
} // namespace geant_exporter
