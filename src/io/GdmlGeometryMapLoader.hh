//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GdmlGeometryMapLoader.hh
//---------------------------------------------------------------------------//
#pragma once

#include "RootLoader.hh"
#include "GdmlGeometryMap.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load GdmlGeometryMap data by reading the imported material and volume data
 * from the ROOT file produced by the app/geant-exporter.
 *
 * \code
 *  GdmlGeometryMapLoader geometry_loader(root_loader);
 *  const auto geometry = geometry_loader();
 * \endcode
 *
 * \sa GdmlGeometryMap
 * \sa RootLoader
 */
class GdmlGeometryMapLoader
{
  public:
    // Construct with RootLoader
    GdmlGeometryMapLoader(RootLoader& root_loader);

    // Return constructed GdmlGeometryMap
    const std::shared_ptr<const GdmlGeometryMap> operator()();

  private:
    RootLoader root_loader_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
