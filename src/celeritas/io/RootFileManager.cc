//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootFileManager.cc
//---------------------------------------------------------------------------//
#include "RootFileManager.hh"

#include <TFile.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with ROOT filename.
 */
RootFileManager::RootFileManager(const char* filename)
{
    CELER_EXPECT(strlen(filename));
    tfile_.reset(TFile::Open(filename, "recreate"));
}

//---------------------------------------------------------------------------//
/*!
 * Write TFile to disk.
 */
void RootFileManager::write()
{
    CELER_EXPECT(tfile_->IsOpen());
    const auto write_status = tfile_->Write();
    CELER_ENSURE(write_status);
}

//---------------------------------------------------------------------------//
/*!
 * Verify if TFile is open.
 */
RootFileManager::operator bool() const
{
    return tfile_->IsOpen();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
