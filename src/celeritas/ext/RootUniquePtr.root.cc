//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootUniquePtr.root.cc
//---------------------------------------------------------------------------//
#include "RootUniquePtr.hh"

#include <TObject.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Call `TObject->Write()` before deletion. Used by TFile and TTree writer
 * classes.
 */
void WriteRootDeleter::operator()(TObject* ptr)
{
    ptr->Write();
    delete ptr;
}

//---------------------------------------------------------------------------//
/*!
 * Deleter only. Used by TFile and TTree reader classes or TObjects that should
 * not invoke `Write()`.
 */
void ReadRootDeleter::operator()(TObject* ptr)
{
    delete ptr;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
