//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImporterInterface.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Construct import data on demand.
 */
class ImporterInterface
{
  public:
    virtual ImportData operator()() = 0;

  protected:
    ImporterInterface() = default;
    CELER_DEFAULT_COPY_MOVE(ImporterInterface);
    ~ImporterInterface() = default;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
