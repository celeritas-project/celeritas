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
 * Load ROOT file as a shared_ptr<TFile>, providing access to the TFile to
 * any class that needs ROOT reading capabilities.
 *
  * \code
 *  RootLoader root_loader("/path/to/root_file.root");
 *  auto tfile = root_loader.get();
 * \endcode
 */
class RootLoader
{
  public:
    // Construct with root filename
    explicit RootLoader(const char* filename);

    // Access the ROOT TFile
    const std::shared_ptr<TFile> get() const;

    // Verify if TFile is open
    explicit operator bool() const;
    
  private:
    std::shared_ptr<TFile> tfile_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
