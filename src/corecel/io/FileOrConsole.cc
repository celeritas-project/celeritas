//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/FileOrConsole.cc
//---------------------------------------------------------------------------//
#include "FileOrConsole.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with filename.
 */
FileOrStdin::FileOrStdin(std::string filename) : filename_{std::move(filename)}
{
    CELER_VALIDATE(!filename_.empty(),
                   << "empty filename is not valid for input");
    if (filename_ == "-")
    {
        filename_ = "<stdin>";
        return;
    }
    // Open the specified file
    inf_.open(filename_);
    CELER_VALIDATE(inf_, << "failed to open '" << filename_ << "'");
}

//---------------------------------------------------------------------------//
/*!
 * Construct with filename.
 */
FileOrStdout::FileOrStdout(std::string filename)
    : filename_{std::move(filename)}
{
    CELER_VALIDATE(!filename_.empty(),
                   << "empty filename is not valid for output");
    if (filename_ == "-")
    {
        filename_ = "<stdout>";
        return;
    }

    // Open the specified file
    outf_.open(filename_);
    CELER_VALIDATE(outf_, << "failed to open '" << filename_ << "'");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
