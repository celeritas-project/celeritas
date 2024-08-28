//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootJsonDumper.cc
//---------------------------------------------------------------------------//
#include "RootJsonDumper.hh"

#include <ostream>
#include <TBufferJSON.h>
#include <TClass.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/io/ImportData.hh"

#include "RootFileManager.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from path to ROOT file.
 */
RootJsonDumper::RootJsonDumper(std::ostream* os) : os_{os}
{
    CELER_EXPECT(os_);
    CELER_VALIDATE(RootFileManager::use_root(),
                   << "cannot interface with ROOT (disabled by user "
                      "environment)");
}

//---------------------------------------------------------------------------//
/*!
 * Write data to the ROOT file.
 */
void RootJsonDumper::operator()(ImportData const& import_data)
{
    CELER_LOG(debug) << "Converting import data to JSON";
    *os_ << TBufferJSON::ConvertToJSON(&import_data,
                                       TClass::GetClass(typeid(import_data)));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
