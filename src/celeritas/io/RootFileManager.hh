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
#include "celeritas/ext/detail/TRootUniquePtr.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * ROOT TFile manager. Currently this class is *not* thread-safe and only works
 * in serial. Since there is only one TFile*, any writer class can just create
 * their own TTrees and ROOT will know how to handle them.
 *
 * If this is expanded to store one TFile per thread, we will need to expand
 * the class to either have a `void make_tree("name, "title", tid)` or provide
 * a `TFile* get(tid)` for direct access (less ideal).
 */
class RootFileManager
{
  public:
    // Construct with filename
    explicit RootFileManager(const char* filename);

    // Write TFile
    void write();

    // Verify if TFile is open
    explicit operator bool() const;

  private:
    using UPTFile = detail::TRootUniquePtr<TFile>;

    UPTFile tfile_;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
RootFileManager::RootFileManager(const char* filename)
{
    CELER_EXPECT(strlen(filename));
    CELER_NOT_CONFIGURED("ROOT");
}

void RootFileManager::write()
{
    CELER_NOT_CONFIGURED("ROOT");
}

RootFileManager::operator bool() const
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
