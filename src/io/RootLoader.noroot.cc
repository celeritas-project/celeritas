//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootLoader.noroot.cc
//---------------------------------------------------------------------------//
#include "RootLoader.hh"

#include "base/Assert.hh"

// We're not linking against ROOT: declare a TFile so that its null-op
// destructor can be called by the shared_ptr destructor.
class TFile
{
};

namespace celeritas
{
//---------------------------------------------------------------------------//
RootLoader::RootLoader(const char*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

auto RootLoader::get() const -> const std::shared_ptr<TFile>
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
