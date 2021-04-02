//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialParamsLoader.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/material/MaterialParams.hh"
#include "RootLoader.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load MaterialParams by reading the imported material data from the ROOT
 * file produced by the app/geant-exporter.
 *
 * \code
 *  MaterialParamsLoader material_loader(root_loader);
 *  const auto material_params = material_loader();
 * \endcode
 *
 * \sa MaterialParams
 * \sa RootLoader
 */
class MaterialParamsLoader
{
  public:
    // Construct with RootLoader
    explicit MaterialParamsLoader(RootLoader& root_loader);

    // Return constructed MaterialParams
    const std::shared_ptr<const MaterialParams> operator()();

  private:
    RootLoader root_loader_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
