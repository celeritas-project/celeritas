//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/RootSafeUniquePtrs.root.cc
//---------------------------------------------------------------------------//
#include "RootSafeUniquePtrs.hh"

#include <TFile.h>
#include <TTree.h>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! TFile deleter
void TFileDeleter::operator()(TFile* ptr) const
{
    ptr->Write();
    delete ptr;
}

//---------------------------------------------------------------------------//
//! TTree deleter
void TTreeDeleter::operator()(TTree* ptr) const
{
    ptr->Write();
    delete ptr;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
