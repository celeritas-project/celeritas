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
 * If this is expanded to store one TFile per thread, we will need to either
 * implement a `void make_tree("name, "title", tid)` or provide a `TFile*
 * get(tid)` for direct access (less ideal).
 */
class RootFileManager
{
  public:
    // Construct with filename
    explicit RootFileManager(const char* filename);

    // Write TFile
    void write();

  public:
    using UPTFile = detail::RootUniquePtr<TFile>;
    UPTFile tfile_;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootFileManager::RootFileManager(const char*)
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
