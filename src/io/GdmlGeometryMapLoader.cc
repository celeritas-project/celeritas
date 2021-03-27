//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GdmlGeometryMapLoader.cc
//---------------------------------------------------------------------------//
#include "GdmlGeometryMapLoader.hh"

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>

#include "base/Assert.hh"
#include "GdmlGeometryMap.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with RootLoader.
 */
GdmlGeometryMapLoader::GdmlGeometryMapLoader(RootLoader& root_loader)
    : root_loader_(root_loader)
{
    CELER_ENSURE(root_loader);
}

//---------------------------------------------------------------------------//
/*!
 * Load GdmlGeometryMap as a shared_ptr.
 */
const std::shared_ptr<const GdmlGeometryMap> GdmlGeometryMapLoader::operator()()
{
    const auto tfile = root_loader_.get();

    // Open geometry branch
    std::unique_ptr<TTree> tree_geometry(tfile->Get<TTree>("geometry"));
    CELER_ASSERT(tree_geometry);
    CELER_ASSERT(tree_geometry->GetEntries() == 1);

    // Load branch and fetch data
    GdmlGeometryMap  geometry;
    GdmlGeometryMap* geometry_ptr = &geometry;

    int err_code
        = tree_geometry->SetBranchAddress("GdmlGeometryMap", &geometry_ptr);
    CELER_ASSERT(err_code >= 0);
    tree_geometry->GetEntry(0);

    return std::make_shared<GdmlGeometryMap>(std::move(geometry));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
