//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootLoader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

class TFile;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load ROOT file as a unique_ptr<TFile> and provide access to it for other
 * loader classes.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    RootLoader root_loader("/path/to/root_file.root");
    auto tfile = root_loader.get();
   \endcode
 */
class RootLoader
{
  public:
    // Construct with defaults
    RootLoader(const char* filename);

    // Access the ROOT TFile
    const std::shared_ptr<TFile> get() const;

    // Verify loader
    operator bool() const;
    
  private:
    std::shared_ptr<TFile> tfile_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
