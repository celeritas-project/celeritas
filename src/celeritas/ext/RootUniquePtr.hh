//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootUniquePtr.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas_config.h"
#include "corecel/Assert.hh"

// Forward-declare ROOT
class TObject;

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Helpers to prevent ROOT from propagating to downstream code.
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Call `TObject->Write()` before deletion. Used by TFile and TTree writer
 * classes.
 */
struct WriteRootDeleter
{
    void operator()(TObject* obj);
};

//---------------------------------------------------------------------------//
/*!
 * Deleter only. Used by TFile and TTree reader classes or TObjects that should
 * not invoke `Write()`.
 */
struct ReadRootDeleter
{
    void operator()(TObject* obj);
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
void WriteRootDeleter::operator()(TObject* ptr)
{
    CELER_NOT_CONFIGURED("ROOT");
}

void ReadRootDeleter::operator()(TObject* ptr)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
