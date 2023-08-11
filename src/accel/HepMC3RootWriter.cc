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
#include "accel/HepMC3RootPrimary.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Convert celeritas::Real3 to std::array<double, 3>.
 */
std::array<double, 3> real3_to_array(Real3 const& src)
{
    std::array<double, 3> dst;
    std::memcpy(&dst, &src, sizeof(src));
    return dst;
}
//---------------------------------------------------------------------------//
}  // namespace

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

    HepMC3RootPrimary write_primary;
    tree->Branch("event_id", &write_primary.event_id);
    tree->Branch("particle", &write_primary.particle);
    tree->Branch("energy", &write_primary.energy);
    tree->Branch("time", &write_primary.time);
    tree->Branch("pos", &write_primary.pos);
    tree->Branch("dir", &write_primary.dir);

    auto read_primaries = reader_();
    while (!read_primaries.empty())
    {
        for (auto const& p : read_primaries)
        {
            write_primary.event_id = p.event_id.get();
            write_primary.particle = 0;  // TODO
            write_primary.energy = p.energy.value();
            write_primary.time = p.time;
            write_primary.pos = real3_to_array(p.position);
            write_primary.dir = real3_to_array(p.direction);
            tree->Fill();
        }
        read_primaries = reader_();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
