//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGdmlLoader.cc
//---------------------------------------------------------------------------//
#include "GeantGdmlLoader.hh"

#include <regex>
#include <G4GDMLAuxStructType.hh>
#include <G4GDMLParser.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4PhysicalVolumeStore.hh>
#include <G4ReflectionFactory.hh>
#include <G4SolidStore.hh>
#include <G4Version.hh>

#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"

#include "ScopedGeantExceptionHandler.hh"
#include "ScopedGeantLogger.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
bool search_pointer(std::string const& s, std::smatch& ptr_match)
{
    // Search either for th end of the expression, or an underscore that likely
    // indicates a _refl or _PV suffix
    static std::regex const ptr_regex{"0x[0-9a-f]{4,16}(?=_|$)"};
    return std::regex_search(s, ptr_match, ptr_regex);
}

//---------------------------------------------------------------------------//
//! Remove pointer address from inside geometry names
template<class StoreT>
void remove_pointers(StoreT& obj_store)
{
    std::smatch sm;
    for (auto* obj : obj_store)
    {
        if (!obj)
            continue;

        std::string const& name = obj->GetName();
        if (search_pointer(name, sm))
        {
            std::string new_name = name.substr(0, sm.position());
            new_name += name.substr(sm.position() + sm.length());
            obj->SetName(new_name);
        }
    }
#if G4VERSION_NUMBER >= 1100
    obj_store.UpdateMap();  // Added by geommng-V10-07-00
#endif
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Load a gdml input file, creating a pointer owned by Geant4.
 *
 * Geant4's constructors for physical/logical volumes register \c this pointers
 * in the "volume stores" which can be cleared with
 * \c celeritas::reset_geant_geometry .
 *
 * Note that material and element names (at least as
 * of Geant4@11.0) are \em always stripped: only volumes and solids keep
 * their extension.
 */
auto GeantGdmlLoader::operator()(std::string const& filename) const -> Result
{
    CELER_EXPECT(!filename.empty());
    CELER_LOG(info) << "Loading Geant4 geometry from GDML at " << filename;

    if (!G4Threading::IsMasterThread())
    {
        // Always-on debug assertion (not a "runtime" error but a
        // subtle programming logic error that always causes a crash)
        CELER_DEBUG_FAIL(
            "Geant4 geometry cannot be loaded from a worker thread", internal);
    }

    ScopedMem record_mem("GeantGdmlLoader.load");
    ScopedTimeLog scoped_time;
    ScopedProfiling profile_this{"geant-gdml-load"};

    ScopedGeantLogger scoped_logger;
    ScopedGeantExceptionHandler scoped_exceptions;

    G4GDMLParser gdml_parser;
    gdml_parser.SetStripFlag(opts_.pointers == PointerTreatment::truncate);

    gdml_parser.Read(filename, /* validate_gdml_schema = */ false);

    Result result;
    result.world = gdml_parser.GetWorldVolume();

    if (opts_.detectors)
    {
        // Find sensitive detectors
        auto const* refl_factory = G4ReflectionFactory::Instance();
        CELER_ASSERT(refl_factory);

        for (auto&& [lv, vecaux] : *gdml_parser.GetAuxMap())
        {
            for (G4GDMLAuxStructType const& aux : vecaux)
            {
                std::string const& sd_name = aux.value;
                if (aux.type == "SensDet")
                {
                    result.detectors.insert({sd_name, lv});
                    if (auto* refl_lv = refl_factory->GetReflectedLV(lv))
                    {
                        // Add the reflected volume as well
                        result.detectors.insert({sd_name, refl_lv});
                    }
                }
            }
        }
    }

    if (opts_.pointers == PointerTreatment::remove)
    {
        remove_pointers(*G4SolidStore::GetInstance());
        remove_pointers(*G4PhysicalVolumeStore::GetInstance());
        remove_pointers(*G4LogicalVolumeStore::GetInstance());
    }

    CELER_ENSURE(result.world);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Write a GDML file to the given filename.
 */
void save_gdml(G4VPhysicalVolume const* world, std::string const& out_filename)
{
    CELER_EXPECT(world);

    CELER_LOG(info) << "Writing Geant4 geometry to GDML at " << out_filename;
    ScopedMem record_mem("save_gdml");
    ScopedTimeLog scoped_time;

    ScopedGeantLogger scoped_logger;
    ScopedGeantExceptionHandler scoped_exceptions;

    G4GDMLParser parser;
    parser.SetOverlapCheck(false);

    if (!world->GetLogicalVolume()->GetRegion())
    {
        CELER_LOG(warning) << "Geant4 regions have not been set up: skipping "
                              "export of energy cuts and regions";
    }
    else
    {
        parser.SetEnergyCutsExport(true);
        parser.SetRegionExport(true);
    }

    parser.SetSDExport(true);
    parser.SetStripFlag(false);
#if G4VERSION_NUMBER >= 1070
    parser.SetOutputFileOverwrite(true);
#endif

    parser.Write(out_filename, world, /* append_pointers = */ true);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
