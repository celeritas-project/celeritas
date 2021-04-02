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
 * Load ImportProcess data by reading the imported process data from the ROOT
 * file produced by the app/geant-exporter.
 *
  * \code
 *  ImportProcessLoader process_loader(root_loader);
 *  const auto processes = process_loader();
 * \endcode
 *
 * \sa ImportProcess
 * \sa RootLoader
 */
class ImportProcessLoader
{
  public:
    // Construct with RootLoader
    explicit ImportProcessLoader(RootLoader& root_loader);

    // Return constructed ImportProcess
    const std::vector<ImportProcess> operator()();

  private:
    RootLoader root_loader_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
