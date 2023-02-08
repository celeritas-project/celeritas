//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
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

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "../GeantConfig.hh"
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
 * Get a multiplicative geant4-natural-units constant to convert the units.
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
        default:
            CELER_ASSERT_UNREACHABLE();
    }
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
#if CELERITAS_G4_V10
        case T_G4PhysicsVector:
            return ImportPhysicsVectorType::unknown;
#endif
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
 * Read values from a Geant4 physics table into an ImportTable.
 */
ImportPhysicsTable
import_table(G4PhysicsTable const& g4table, ImportTableType table_type)
{
    CELER_EXPECT(table_type != ImportTableType::size_);
    ImportPhysicsTable table;
    table.table_type = table_type;
    switch (table_type)
    {
        case ImportTableType::dedx:
        case ImportTableType::dedx_process:
        case ImportTableType::dedx_subsec:
        case ImportTableType::dedx_unrestricted:
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
        default:
            CELER_ASSERT_UNREACHABLE();
    };

    // Convert units
    double x_scaling = units_to_scaling(table.x_units);
    double y_scaling = units_to_scaling(table.y_units);

    // Save physics vectors
    for (auto const* g4vector : g4table)
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

    return table;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with a selected list of tables.
 */
GeantProcessImporter::GeantProcessImporter(
    TableSelection which_tables,
    std::vector<ImportMaterial> const& materials,
    std::vector<ImportElement> const& elements)
    : materials_(materials), elements_(elements), which_tables_(which_tables)
{
    CELER_ENSURE(!materials_.empty());
    CELER_ENSURE(!elements_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
GeantProcessImporter::~GeantProcessImporter() = default;

//---------------------------------------------------------------------------//
/*!
 * Load and return physics tables from a given particle and process.
 *
 * If the process was already returned, \c operator() will return an
 * empty object.
 */
ImportProcess
GeantProcessImporter::operator()(G4ParticleDefinition const& particle,
                                 G4VProcess const& process)
{
    // Check for duplicate processes
    auto [prev, inserted] = written_processes_.insert({&process, {&particle}});

    if (!inserted)
    {
        static const celeritas::TypeDemangler<G4VProcess> demangle_process;
        CELER_LOG(debug) << "Skipping process '" << process.GetProcessName()
                         << "' (RTTI: " << demangle_process(process)
                         << ") for particle " << particle.GetParticleName()
                         << ": duplicate of particle "
                         << prev->second.particle->GetParticleName();
        return {};
    }
    CELER_LOG(debug) << "Saving process '" << process.GetProcessName()
                     << "' for particle " << particle.GetParticleName() << " ("
                     << particle.GetPDGEncoding() << ')';

    // Save process and particle info
    process_ = {};
    process_.process_type = to_import_process_type(process.GetProcessType());
    process_.process_class = to_import_process_class(process);
    process_.particle_pdg = particle.GetPDGEncoding();

    if (auto const* em_process = dynamic_cast<G4VEmProcess const*>(&process))
    {
        this->store_em_process(*em_process);
    }
    else if (auto const* energy_loss
             = dynamic_cast<G4VEnergyLossProcess const*>(&process))
    {
        this->store_eloss_process(*energy_loss);
    }
    else if (auto const* multiple_scattering
             = dynamic_cast<G4VMultipleScattering const*>(&process))
    {
        this->store_msc_process(*multiple_scattering);
    }
    else
    {
        static const celeritas::TypeDemangler<G4VProcess> demangle_process;
        CELER_LOG(error) << "Cannot export unknown process '"
                         << process.GetProcessName()
                         << "' (RTTI: " << demangle_process(process) << ")";
    }

    CELER_ENSURE(process_);
    CELER_ENSURE(std::all_of(
        process_.models.begin(),
        process_.models.end(),
        [](ImportModel const& m) { return static_cast<bool>(m); }));
    return std::move(process_);
}

//---------------------------------------------------------------------------//
// PRIVATE
//---------------------------------------------------------------------------//
/*!
 * Store common properties of the current process.
 *
 * As of Geant4 11, these functions are non-virtual functions of the daughter
 * classes that have the same interface.
 */
template<class T>
void GeantProcessImporter::store_common_process(T const& process)
{
    static_assert(std::is_base_of<G4VProcess, T>::value,
                  "process must be a G4VProcess");

    // Save secondaries
    if (auto const* secondary = process.SecondaryParticle())
    {
        process_.secondary_pdg = secondary->GetPDGEncoding();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Store EM cross section tables for the current process.
 */
void GeantProcessImporter::store_em_process(G4VEmProcess const& process)
{
    this->store_common_process(process);

    GeantModelImporter convert_model(materials_,
                                     PDGNumber{process_.particle_pdg},
                                     PDGNumber{process_.secondary_pdg});
#if CELERITAS_G4_V10
    for (auto i : celeritas::range(process.GetNumberOfModels()))
#else
    for (auto i : celeritas::range(process.NumberOfModels()))
#endif
    {
        process_.models.push_back(convert_model(*process.GetModelByIndex(i)));
        CELER_ASSERT(process_.models.back());
    }

    // Save cross section tables if available
    this->add_table(process.LambdaTable(), ImportTableType::lambda);
    this->add_table(process.LambdaTablePrim(), ImportTableType::lambda_prim);
}

//---------------------------------------------------------------------------//
/*!
 * Store energy loss XS tables to this->process_.
 *
 * The following XS tables do not exist in Geant4 v11:
 * - DEDXTableForSubsec()
 * - IonisationTableForSubsec()
 * - SubLambdaTable()
 */
void GeantProcessImporter::store_eloss_process(
    G4VEnergyLossProcess const& process)
{
    this->store_common_process(process);

    // Note: NumberOfModels/GetModelByIndex is a *not* a virtual method on
    // G4VProcess, so this loop cannot yet be combined with the one in
    // store_em_process .
    // TODO: when we drop support for Geant4 10 we can use a template to
    // move this into store_common_process...

    GeantModelImporter convert_model(materials_,
                                     PDGNumber{process_.particle_pdg},
                                     PDGNumber{process_.secondary_pdg});
    for (auto i : celeritas::range(process.NumberOfModels()))
    {
        process_.models.push_back(convert_model(*process.GetModelByIndex(i)));
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
        // Secondary range is removed in 11.1
        this->add_table(process.SecondaryRangeTable(),
                        ImportTableType::secondary_range);
#endif

        this->add_table(process.DEDXunRestrictedTable(),
                        ImportTableType::dedx_unrestricted);
        this->add_table(process.CSDARangeTable(), ImportTableType::csda_range);
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
void GeantProcessImporter::store_msc_process(G4VMultipleScattering const& process)
{
    GeantModelImporter convert_model(
        materials_, PDGNumber{process_.particle_pdg}, PDGNumber{});

#if CELERITAS_G4_V10
    for (auto i : celeritas::range(4))
#else
    for (auto i : celeritas::range(process.NumberOfModels()))
#endif
    {
        if (G4VEmModel* model = process.GetModelByIndex(i))
        {
            process_.models.push_back(convert_model(*model));
            this->add_table(model->GetCrossSectionTable(),
                            ImportTableType::lambda);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write data from a Geant4 physics table if available.
 */
void GeantProcessImporter::add_table(G4PhysicsTable const* g4table,
                                     ImportTableType table_type)
{
    if (!g4table)
    {
        // Table isn't present
        return;
    }

    // Check for duplicate tables
    auto [prev, inserted] = written_tables_.insert(
        {g4table, {process_.particle_pdg, process_.process_class, table_type}});

    CELER_LOG(debug) << (inserted ? "Saving" : "Skipping duplicate")
                     << " physics table " << process_.particle_pdg << '.'
                     << to_cstring(process_.process_class) << '.'
                     << to_cstring(table_type);
    if (!inserted)
    {
        CELER_LOG(debug) << "PreviousExisting table was at"
                         << prev->second.particle_pdg << '.'
                         << to_cstring(prev->second.process_class) << '.'
                         << to_cstring(prev->second.table_type);
        return;
    }

    process_.tables.push_back(import_table(*g4table, table_type));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
