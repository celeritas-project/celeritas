//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/HepMC3RootWriter.cc
//---------------------------------------------------------------------------//
#include "HepMC3RootWriter.hh"

#include <TTree.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "orange/Types.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/phys/Primary.hh"
#include "accel/HepMC3RootEvent.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with HepMC3 input filename.
 */
HepMC3RootWriter::HepMC3RootWriter(std::string const& hepmc3_input_name)
    : reader_(hepmc3_input_name)
{
}

//---------------------------------------------------------------------------//
/*!
 * Export HepMC3 primary data to ROOT.
 */
void HepMC3RootWriter::operator()(std::string const& root_output_name)
{
    CELER_LOG(info) << "Generating \'" << root_output_name
                    << "\' with list of primaries from the HepMC3 input";

    RootFileManager root_mgr(root_output_name.c_str());
    auto tree = root_mgr.make_tree(this->tree_name(), this->tree_name());

    HepMC3RootEvent event;
    tree->Branch("event_id", &event.event_id);
    tree->Branch("particle", &event.particle);
    tree->Branch("energy", &event.energy);
    tree->Branch("time", &event.time);
    tree->Branch("pos_x", &event.pos_x);
    tree->Branch("pos_y", &event.pos_y);
    tree->Branch("pos_z", &event.pos_z);
    tree->Branch("dir_x", &event.dir_x);
    tree->Branch("dir_y", &event.dir_y);
    tree->Branch("dir_z", &event.dir_z);

    auto primaries = reader_();
    while (!primaries.empty())
    {
        event.event_id = primaries.front().event_id.unchecked_get();

        // TODO: Resize vectors before loop?
        for (auto const& hepmc3_prim : primaries)
        {
            event.particle.push_back(/* pdg */ 0);  // TODO
            event.energy.push_back(hepmc3_prim.energy.value());
            event.time.push_back(hepmc3_prim.time);
            event.pos_x.push_back(hepmc3_prim.position[0]);
            event.pos_y.push_back(hepmc3_prim.position[1]);
            event.pos_z.push_back(hepmc3_prim.position[2]);
            event.dir_x.push_back(hepmc3_prim.direction[0]);
            event.dir_y.push_back(hepmc3_prim.direction[1]);
            event.dir_z.push_back(hepmc3_prim.direction[2]);
        }
        tree->Fill();
        event = HepMC3RootEvent();
        primaries = reader_();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
