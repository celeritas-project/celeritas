//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffParamsLoader.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/CutoffParams.hh"
#include "RootLoader.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load CutoffParams by reading the imported production cut data from the ROOT
 * file produced by the app/geant-exporter.
 *
 * \code
 *  CutoffParamsLoader cutoff_loader(root_loader);
 *  const auto Cutoff_params = cutoff_loader();
 * \endcode
 *
 * \sa RootLoader
 */
class CutoffParamsLoader
{
  public:
    // Construct with RootLoader
    CutoffParamsLoader(RootLoader& root_loader);

    // Return constructed CutoffParams
    const std::shared_ptr<const CutoffParams> operator()();

  private:
    RootLoader root_loader_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
