//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootImporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "celeritas/io/ImportData.hh"

#include "detail/RootUniquePtr.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Create an \c ImportData object from a ROOT data file.
 *
 * RootImporter loads particle, element, material, process, and volume
 * information from a ROOT file that contains an \c ImportData object.
 * Currently, said ROOT file is created by the \c RootExporter class.
 *
 * \c RootImporter , along with all \c Import[Class] type of classes, are the
 * link between Geant4 and Celeritas. Every Celeritas' host/device class that
 * relies on imported data has its own \c from_import(...) function that will
 * take the data loaded by the \c RootImporter and load it accordingly:
 *
 * \code
 *  RootImporter import("/path/to/root_file.root");
 *  const auto data            = import();
 *  const auto particle_params = ParticleParams::from_import(data);
 *  const auto material_params = MaterialParams::from_import(data);
 *  const auto cutoff_params   = CutoffParams::from_import(data);
 *  // And so on
 * \endcode
 */
class RootImporter
{
  public:
    // Construct with ROOT file name
    explicit RootImporter(const char* filename);

    // Load data from the ROOT files
    ImportData operator()();

  private:
    // ROOT file
    detail::RootUniquePtr<TFile> root_input_;

    // ROOT TTree name
    static const char* tree_name();
    // ROOT TBranch name
    static const char* branch_name();
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootImporter::RootImporter(const char*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline auto RootImporter::operator()() -> ImportData
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
