//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootExporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Assert.hh"

#include "RootUniquePtr.hh"

// Forward declare ROOT
class TFile;

namespace celeritas
{
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Write an \c ImportData object to a ROOT file.
 *
 * \code
 *  RootExporter export("/path/to/root_file.root");
 *  export(my_import_data);
 * \endcode
 */
class RootExporter
{
  public:
    // Construct with ROOT file name
    explicit RootExporter(char const* filename);

    // Save data to the ROOT file
    void operator()(ImportData const& data);

  private:
    UPRootFileWritable root_output_;

  private:
    // ROOT TTree name
    static char const* tree_name();
    // ROOT TBranch name
    static char const* branch_name();
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootExporter::RootExporter(char const*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void RootExporter::operator()(ImportData const&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
