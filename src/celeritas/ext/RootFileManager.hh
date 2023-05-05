//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootFileManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "celeritas/ext/RootUniquePtr.hh"

// Forward declare ROOT
class TFile;
class TTree;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * ROOT TFile manager. Currently this class is *not* thread-safe. Since there
 * is only one TFile*, any writer class (such as `RootStepWriter.hh`) can just
 * create their own TTrees and ROOT will know how to handle them.
 *
 * If this is expanded to store one TFile per thread, we will need to expand
 * `make_tree("name, "title")` to include a thread id as input parameter.
 */
class RootFileManager
{
  public:
    // Construct with filename
    explicit RootFileManager(char const* filename);

    // Create tree by passing a name and title
    UPRootWritable<TTree> make_tree(char const* name, char const* title);

    // Manually write and close TFile
    void close();

  public:
    UPRootWritable<TFile> tfile_;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootFileManager::RootFileManager(char const*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline UPRootWritable<TTree>
RootFileManager::make_tree(char const*, char const*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void RootFileManager::close()
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
