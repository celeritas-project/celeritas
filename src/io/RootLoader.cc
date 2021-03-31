//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootLoader.cc
//---------------------------------------------------------------------------//
#include "RootLoader.hh"

#include <TFile.h>

#include "base/Assert.hh"
#include "comm/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with path and name of the ROOT file.
 */
RootLoader::RootLoader(const char* filename)
{
    CELER_LOG(status) << "Opening ROOT file";
    tfile_ = std::shared_ptr<TFile>(TFile::Open(filename, "read"));

    CELER_VALIDATE(tfile_ && !tfile_->IsZombie(),
                   "Could not read ROOT file '" << filename << "'");
}

//---------------------------------------------------------------------------//
/*!
 * Access ROOT TFile.
 */
const std::shared_ptr<TFile> RootLoader::get()
{
    CELER_EXPECT(tfile_);
    return tfile_;
}

//---------------------------------------------------------------------------//
/*!
 * Verify if TFile is open. Mostly used by assertion macros.
 */
RootLoader::operator bool() const
{
    return tfile_->IsOpen();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
