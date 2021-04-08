//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootImporter.noroot.cc
//---------------------------------------------------------------------------//
#include "RootImporter.hh"

#include "base/Assert.hh"

// We're not linking against ROOT: declare a TFile so that its null-op
// destructor can be called by the unique_ptr destructor.
class TFile
{
};

namespace celeritas
{
struct ImportData
{
};
//---------------------------------------------------------------------------//
RootImporter::RootImporter(const char*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

RootImporter::~RootImporter() = default;

auto RootImporter::operator()(const char* tree_name, const char* branch_name)
    -> ImportData
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
