//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootFileManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Assert.hh"

#include "RootUniquePtr.hh"

// Forward declare ROOT
class TFile;
class TTree;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * ROOT TFile manager.
 *
 * Currently this class is *not* thread-safe. Since there
 * is only one TFile*, any writer class (such as `RootStepWriter.hh`) can just
 * create their own TTrees and ROOT will know how to handle them.
 *
 * If this is expanded to store one TFile per thread, we will need to expand
 * `make_tree("name, "title")` to include a thread id as input parameter.
 */
class RootFileManager
{
  public:
#if CELERITAS_USE_ROOT
    // Whether ROOT output is enabled
    static bool use_root();
#else
    // ROOT is never enabled if ROOT isn't available
    constexpr static bool use_root() { return false; }
#endif

    // Construct with filename
    explicit RootFileManager(char const* filename);

    // Get the ROOT filename
    char const* filename() const;

    // Create tree by passing a name and title
    UPRootTreeWritable make_tree(char const* name, char const* title);

    // Manually write TFile
    void write();

  public:
    UPRootFileWritable tfile_;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootFileManager::RootFileManager(char const*)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
