//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportProcessLoader.hh
//---------------------------------------------------------------------------//
#pragma once

#include "RootLoader.hh"
#include "ImportProcess.hh"
#include <vector>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load ImportProcess data by reading the imported material data from the ROOT
 * file produced by the app/geant-exporter.
 *
  * \code
 *  ImportProcessLoader process_loader(root_loader);
 *  const auto processes = process_loader();
 * \endcode
 */
class ImportProcessLoader
{
  public:
    // Construct with RootLoader
    ImportProcessLoader(RootLoader& root_loader);

    // Return constructed MaterialParams
    const std::vector<const ImportProcess> operator()();

  private:
    RootLoader root_loader_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
