//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/HepMC3RootWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <array>

#include "corecel/Macros.hh"
#include "celeritas/io/EventReader.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Export HepMC3 primary data to ROOT.
 *
 * One TTree entry represents one event.
 */
class HepMC3RootWriter
{
  public:
    // Construct with input HepMC3 filename
    explicit HepMC3RootWriter(std::string const& hepmc3_input_name);

    // Export HepMC3 primary data to ROOT
    void operator()(std::string const& root_output_name);

  private:
    EventReader reader_;

  private:
    // Hardcoded TTree name and title
    char const* tree_name() { return "primaries"; }
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_HEPMC3
inline Hepmc3RootWriter(std::string const&)
{
    (void)sizeof(reader_);
    CELER_NOT_CONFIGURED("HepMC3");
}
#endif

#if !CELERITAS_USE_ROOT
inline void HepMC3RootWriter::operator()(std::string const&)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
