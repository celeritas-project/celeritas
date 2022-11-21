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
 * Destruct by invoking TFile Write() and Close() if not done so previously.
 */
RootFileManager::~RootFileManager()
{
    if (tfile_->IsOpen())
    {
        this->close();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write and close the TFile before destruction.
 */
void RootFileManager::close()
{
    CELER_EXPECT(tfile_->IsOpen());
    const auto write_status = tfile_->Write();
    CELER_ENSURE(!write_status);
    tfile_->Close();
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
