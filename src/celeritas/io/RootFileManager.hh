//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootFileManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "celeritas/ext/detail/RootUniquePtr.hh"

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
    explicit RootFileManager(const char* filename);

    // Create tree by passing a name and title
    detail::RootUniquePtr<TTree> make_tree(const char* name, const char* title);

    // Write TFile
    void write();

  public:
    detail::RootUniquePtr<TFile> tfile_;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootFileManager::RootFileManager(const char*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline detail::RootUniquePtr<TTree>
RootFileManager::make_tree(const char*, const char*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void RootFileManager::write()
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
