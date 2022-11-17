//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootFileManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "celeritas/ext/detail/TRootUniquePtr.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * ROOT TFile manager.
 */
class RootFileManager
{
  public:
    //!@{
    //! \name Type aliases
    using UPTFile = detail::TRootUniquePtr<TFile>;
    //!@}

    // Construct with filename
    explicit RootFileManager(const char* filename);

    // Write and close TFile (if still open) at destruction time
    ~RootFileManager();

    // Write and close TFile before destruction
    void close();

    // Verify if tfile is open
    explicit operator bool() const;

  private:
    UPTFile tfile_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
