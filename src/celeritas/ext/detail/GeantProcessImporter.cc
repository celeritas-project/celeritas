//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantProcessImporter.cc
//---------------------------------------------------------------------------//
#include "GeantProcessImporter.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4PhysicsTable.hh>
#include <G4PhysicsVector.hh>
#include <G4PhysicsVectorType.hh>
#include <G4ProcessType.hh>
#include <G4String.hh>
#include <G4VEmProcess.hh>
#include <G4VEnergyLossProcess.hh>
#include <G4VMultipleScattering.hh>
#include <G4VProcess.hh>
#include <G4Version.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "GeantModelImporter.hh"

using CLHEP::cm;
using CLHEP::cm2;
using CLHEP::MeV;

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Convert process type from Geant4 to Celeritas IO.
 */
ImportProcessType to_import_process_type(G4ProcessType g4_process_type)
{
    switch (g4_process_type)
    {
        case G4ProcessType::fNotDefined:
            return ImportProcessType::other;
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
ImportProcessClass to_import_process_class(G4VProcess const& process)
{
    auto&& name = process.GetProcessName();
    ImportProcessClass result;
    try
    {
        result = geant_name_to_import_process_class(name);
    }
    catch (celeritas::RuntimeError const&)
    {
        CELER_LOG(warning) << "Encountered unknown process '" << name << "'";
        result = ImportProcessClass::other;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize a process result.
 */
ImportProcess
init_process(G4ParticleDefinition const& particle, G4VProcess const& process)
{
    CELER_LOG(debug) << "Saving process '" << process.GetProcessName()
                     << "' for particle " << particle.GetParticleName() << " ("
                     << particle.GetPDGEncoding() << ')';

    ImportProcess result;
    result = {};
    result.process_type = to_import_process_type(process.GetProcessType());
    result.process_class = to_import_process_class(process);
    result.particle_pdg = particle.GetPDGEncoding();

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the PDG of a process.
 */
template<class T>
int get_secondary_pdg(T const& process)
{
    static_assert(std::is_base_of<G4VProcess, T>::value,
                  "process must be a G4VProcess");

    // Save secondaries
    if (auto const* secondary = process.SecondaryParticle())
    {
        return secondary->GetPDGEncoding();
    }
    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * Get a multiplicative geant4-natural-units constant to convert the units.
 */
double units_to_scaling(ImportUnits units)
{
    switch (units)
    {
        case ImportUnits::none:
            return 1;
        case ImportUnits::mev:
            return 1 / MeV;
        case ImportUnits::mev_per_cm:
            return cm / MeV;
        case ImportUnits::cm:
            return 1 / cm;
        case ImportUnits::cm_inv:
            return cm;
        case ImportUnits::cm_mev_inv:
            return cm * MeV;
        case ImportUnits::mev_2_per_cm:
            return cm / (MeV * MeV);
        case ImportUnits::cm_2:
            return 1 / (cm * cm);
        case ImportUnits::size_:
            CELER_ASSERT_UNREACHABLE();
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Convert physics vector type from Geant4 to Celeritas IO.
 *
 * Geant4 v11 has a different set of G4PhysicsVectorType enums.
 */
ImportPhysicsVectorType
to_import_physics_vector_type(G4PhysicsVectorType g4_vector_type)
{
    switch (g4_vector_type)
    {
#if G4VERSION_NUMBER < 1100
        case T_G4PhysicsVector:
            return ImportPhysicsVectorType::unknown;
#endif
        case T_G4PhysicsLinearVector:
            return ImportPhysicsVectorType::linear;
        case T_G4PhysicsLogVector:
#if G4VERSION_NUMBER < 1100
        case T_G4PhysicsLnVector:
#endif
            return ImportPhysicsVectorType::log;
        case T_G4PhysicsFreeVector:
#if G4VERSION_NUMBER < 1100
        case T_G4PhysicsOrderedFreeVector:
        case T_G4LPhysicsFreeVector:
#endif
            return ImportPhysicsVectorType::free;
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Import data from a Geant4 physics table if available.
 */
void append_table(G4PhysicsTable const* g4table,
                  ImportTableType table_type,
                  std::vector<ImportPhysicsTable>* tables)
{
    if (!g4table)
    {
        // Table isn't present
        return;
    }

    CELER_EXPECT(table_type != ImportTableType::size_);
    ImportPhysicsTable table;
    table.table_type = table_type;
    switch (table_type)
    {
        case ImportTableType::dedx:
            table.x_units = ImportUnits::mev;
            table.y_units = ImportUnits::mev_per_cm;
            break;
        case ImportTableType::range:
            table.x_units = ImportUnits::mev;
            table.y_units = ImportUnits::cm;
            break;
        case ImportTableType::lambda:
            table.x_units = ImportUnits::mev;
            table.y_units = ImportUnits::cm_inv;
            break;
        case ImportTableType::lambda_prim:
            table.x_units = ImportUnits::mev;
            table.y_units = ImportUnits::cm_mev_inv;
            break;
        case ImportTableType::msc_xs:
            table.x_units = ImportUnits::mev;
            table.y_units = ImportUnits::mev_2_per_cm;
            break;
        default:
            CELER_ASSERT_UNREACHABLE();
    };

    // Convert units
    double x_scaling = units_to_scaling(table.x_units);
    double y_scaling = units_to_scaling(table.y_units);

    // Save physics vectors
    for (auto const* g4vector : *g4table)
    {
        ImportPhysicsVector import_vec;

        // Populate ImportPhysicsVectors
        import_vec.vector_type
            = to_import_physics_vector_type(g4vector->GetType());
        import_vec.x.reserve(g4vector->GetVectorLength());
        import_vec.y.reserve(import_vec.x.size());

        for (auto j : celeritas::range(g4vector->GetVectorLength()))
        {
            import_vec.x.push_back(g4vector->Energy(j) * x_scaling);
            import_vec.y.push_back((*g4vector)[j] * y_scaling);
        }
        table.physics_vectors.push_back(std::move(import_vec));
    }

    tables->push_back(std::move(table));
}

template<class T>
bool all_are_assigned(std::vector<T> const& arr)
{
    return std::all_of(arr.begin(), arr.end(), [](T const& v) {
        return static_cast<bool>(v);
    });
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with a selected list of tables.
 */
GeantProcessImporter::GeantProcessImporter(
    TableSelection,
    std::vector<ImportMaterial> const& materials,
    std::vector<ImportElement> const& elements)
    : materials_(materials), elements_(elements)
{
    CELER_ENSURE(!materials_.empty());
    CELER_ENSURE(!elements_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Store EM cross section tables for the current process.
 *
 * Cross sections are calculated in G4EmModelManager::FillLambdaVector by
 * calling G4VEmModel::CrossSection .
 */
ImportProcess
GeantProcessImporter::operator()(G4ParticleDefinition const& particle,
                                 G4VEmProcess const& process)
{
    auto result = init_process(particle, process);
    result.secondary_pdg = get_secondary_pdg(process);

    GeantModelImporter convert_model(materials_,
                                     PDGNumber{result.particle_pdg},
                                     PDGNumber{result.secondary_pdg});
#if G4VERSION_NUMBER < 1100
    for (auto i : celeritas::range(process.GetNumberOfModels()))
#else
    for (auto i : celeritas::range(process.NumberOfModels()))
#endif
    {
        result.models.push_back(convert_model(*process.GetModelByIndex(i)));
        CELER_ASSERT(result.models.back());
    }

    // Save cross section tables if available
    append_table(
        process.LambdaTable(), ImportTableType::lambda, &result.tables);
    append_table(process.LambdaTablePrim(),
                 ImportTableType::lambda_prim,
                 &result.tables);
    CELER_ENSURE(result && all_are_assigned(result.models));
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Store energy loss XS tables to this->result.
 *
 * The following XS tables do not exist in Geant4 v11:
 * - DEDXTableForSubsec()
 * - IonisationTableForSubsec()
 * - SubLambdaTable()
 */
ImportProcess
GeantProcessImporter::operator()(G4ParticleDefinition const& particle,
                                 G4VEnergyLossProcess const& process)
{
    auto result = init_process(particle, process);
    result.secondary_pdg = get_secondary_pdg(process);

    // Note: NumberOfModels/GetModelByIndex is a *not* a virtual method on
    // G4VProcess.

    GeantModelImporter convert_model(materials_,
                                     PDGNumber{result.particle_pdg},
                                     PDGNumber{result.secondary_pdg});
    for (auto i : celeritas::range(process.NumberOfModels()))
    {
        result.models.push_back(convert_model(*process.GetModelByIndex(i)));
    }

    if (process.IsIonisationProcess())
    {
        // The de/dx and range tables created by summing the contribution from
        // each energy loss process are stored in the "ionization process"
        // (which might be ionization or might be another arbitrary energy loss
        // process if there is no ionization in the problem).
        append_table(
            process.DEDXTable(), ImportTableType::dedx, &result.tables);
        append_table(process.RangeTableForLoss(),
                     ImportTableType::range,
                     &result.tables);
    }

    append_table(
        process.LambdaTable(), ImportTableType::lambda, &result.tables);

    CELER_ENSURE(result && all_are_assigned(result.models));
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Store multiple scattering XS tables to the process data.
 *
 * Whereas other EM processes combine the model tables into a single process
 * table, MSC keeps them independent.
 *
 * Starting on Geant4 v11, G4MultipleScattering provides \c NumberOfModels() .
 *
 * The cross sections are stored with an extra factor of E^2 multiplied in.
 * They're calculated in G4LossTableBuilder::BuildTableForModel which calls
 * G4VEmModel::Value.
 */
std::vector<ImportMscModel>
GeantProcessImporter::operator()(G4ParticleDefinition const& particle,
                                 G4VMultipleScattering const& process)
{
    std::vector<ImportMscModel> result;
    int primary_pdg = particle.GetPDGEncoding();

    GeantModelImporter convert_model(
        materials_, PDGNumber{primary_pdg}, PDGNumber{});
    std::vector<ImportPhysicsTable> temp_tables;

#if G4VERSION_NUMBER < 1100
    for (auto i : celeritas::range(4))
#else
    for (auto i : celeritas::range(process.NumberOfModels()))
#endif
    {
        if (G4VEmModel* model = process.GetModelByIndex(i))
        {
            ImportMscModel imm;
            imm.particle_pdg = primary_pdg;
            try
            {
                imm.model_class
                    = geant_name_to_import_model_class(model->GetName());
            }
            catch (celeritas::RuntimeError const&)
            {
                CELER_LOG(warning) << "Encountered unknown process '"
                                   << model->GetName() << "'";
                imm.model_class = ImportModelClass::other;
            }
            append_table(model->GetCrossSectionTable(),
                         ImportTableType::msc_xs,
                         &temp_tables);
            CELER_EXPECT(temp_tables.size() == 1);
            imm.xs_table = std::move(temp_tables.back());
            temp_tables.clear();
            result.push_back(std::move(imm));
        }
    }

    CELER_ENSURE(all_are_assigned(result));
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
