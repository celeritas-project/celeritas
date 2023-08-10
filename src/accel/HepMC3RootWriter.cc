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

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Copy a `celeritas::Real3` to an `std::array<double, 3>`.
 * TODO: generalize this if it's used more than here and RSW?
 */
void real3_to_array(Real3 const& src, std::array<double, 3>& dst)
{
    std::memcpy(&dst, &src, sizeof(src));
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
    RootFileManager root_mgr(root_output_name.c_str());
    auto tree = root_mgr.make_tree("primaries", "primaries");

    Primary primary;
    tree->Branch("event_id", &primary.event_id);
    tree->Branch("energy", &primary.energy);
    tree->Branch("time", &primary.time);
    tree->Branch("pos", &primary.pos);
    tree->Branch("dir", &primary.dir);

    auto primaries = reader_();
    while (!primaries.empty())
    {
        for (auto const& hepmc3_prim : primaries)
        {
            primary.event_id = hepmc3_prim.event_id.unchecked_get();
            primary.energy = hepmc3_prim.energy.value();
            primary.time = hepmc3_prim.time;
            real3_to_array(hepmc3_prim.position, primary.pos);
            real3_to_array(hepmc3_prim.direction, primary.dir);
            tree->Fill();
        }
        primaries = reader_();
    }

    CELER_LOG(info) << "Generated \'" << root_output_name
                    << "\' with list of primaries from the HepMC3 input";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
