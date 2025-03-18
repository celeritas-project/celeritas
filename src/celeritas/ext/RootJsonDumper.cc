//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * Construct with an output stream.
 */
RootJsonDumper::RootJsonDumper(std::ostream& os) : os_{&os}
{
    CELER_VALIDATE(
        RootFileManager::use_root(),
        << R"(cannot interface with ROOT (disabled by user environment))");
}

//---------------------------------------------------------------------------//
/*!
 * Write JSON-formatted data to the stream.
 */
void RootJsonDumper::operator()(ImportData const& import_data)
{
    CELER_LOG(debug) << "Converting import data to JSON";
    *os_ << TBufferJSON::ConvertToJSON(&import_data,
                                       TClass::GetClass(typeid(import_data)));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
