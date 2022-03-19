//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TFileUniquePtr.cc
//---------------------------------------------------------------------------//
#include "TFileUniquePtr.hh"

#include <TFile.h>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
void TFileDeleter::operator()(TFile* ptr) const
{
    delete ptr;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
