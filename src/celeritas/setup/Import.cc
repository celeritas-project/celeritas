//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Import.cc
//---------------------------------------------------------------------------//
#include "Import.hh"

#include <map>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/io/AtomicRelaxationReader.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportElement.hh"
#include "celeritas/io/LivermorePEReader.hh"
#include "celeritas/io/SeltzerBergerReader.hh"

namespace celeritas
{
namespace setup
{
namespace
{
//---------------------------------------------------------------------------//
//! Generate a map of read data for all loaded elements.
class AllElementReader
{
  public:
    //!@{
    //! \name Type aliases
    using VecElements = std::vector<ImportElement>;
    //!@}

  public:
    //! Construct from vector of imported elements
    explicit AllElementReader(VecElements const& els) : elements_(els)
    {
        CELER_EXPECT(!elements_.empty());
    }

    //! Load a map of data for all stored elements
    template<class ReadOneElement>
    auto operator()(ReadOneElement&& read_el) const -> decltype(auto)
    {
        using result_type = typename ReadOneElement::result_type;

        std::map<AtomicNumber, result_type> result_map;

        for (ImportElement const& element : elements_)
        {
            AtomicNumber z{element.atomic_number};
            CELER_ASSERT(z);
            result_map.insert({z, read_el(z)});
        }
        return result_map;
    }

  private:
    std::vector<ImportElement> const& elements_;
};

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Load all physics data from a ROOT file.
 */
void physics_from(inp::PhysicsFromFile const& pff, ImportData& imported)
{
    ScopedProfiling profile_this{"load-physics-root"};

    CELER_VALIDATE(!pff.input.empty(), << "no file import specified");
    // Import all physics data from ROOT file
    imported = RootImporter(pff.input)();
}

//---------------------------------------------------------------------------//
/*!
 * Load from Geant4 in memory.
 */
void physics_from(inp::PhysicsFromGeant const& pfg, ImportData& imported)
{
    ScopedProfiling profile_this{"load-physics-geant"};
    imported = GeantImporter{}(pfg.data_selection);
}

//---------------------------------------------------------------------------//
/*!
 * Load from Geant4 data files.
 *
 * Based on what elements and processes are in the import data, this will load
 * data from the input physics files.
 */
void physics_from(inp::PhysicsFromGeantFiles const& pfgf, ImportData& imported)
{
    CELER_EXPECT(!imported.elements.empty());

    ScopedProfiling profile_this{"load-physics-files"};

    AllElementReader load_data{imported.elements};
    auto have_process = [&imported](ImportProcessClass ipc) {
        return std::any_of(imported.processes.begin(),
                           imported.processes.end(),
                           [ipc](ImportProcess const& ip) {
                               return ip.process_class == ipc;
                           });
    };

    //// PHOTOELECTRIC ////

    if (have_process(ImportProcessClass::e_brems))
    {
        inp::SeltzerBergerModel sb_model;
        sb_model.atomic_xs = load_data(SeltzerBergerReader{});
        imported.seltzer_berger = std::move(sb_model);
    }
    if (have_process(ImportProcessClass::photoelectric))
    {
        // TODO: delete interpolation parameter; see PhysicsFromGeant docs
        inp::LivermorePhotoModel lp_model;
        lp_model.atomic_xs
            = load_data(LivermorePEReader{imported.em_params.interpolation});
        imported.livermore_photo = std::move(lp_model);
    }

    //// NEUTRONS ////

    if (!pfgf.neutron_dir.empty())
    {
        // TODO: see ProcessBuilder::build_neutron_elastic
        CELER_LOG(warning) << "Ignoring PhysicsFromGeantFiles.neutron_dir";
    }

    //// ATOMIC RELAXATION ////

    if (imported.em_params.fluorescence)
    {
        // TODO: split loading between fluorescence and auger
        inp::AtomicRelaxation ar_process;
        ar_process.atomic_xs = load_data(AtomicRelaxationReader{});
        imported.atomic_relaxation = std::move(ar_process);
    }
    else if (imported.em_params.auger)
    {
        CELER_LOG(warning) << "Auger emission is ignored because "
                              "fluorescent atomic relaxation is disabled";
    }
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
