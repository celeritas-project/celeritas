//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/FileOrConsole.cc
//---------------------------------------------------------------------------//
#include "FileOrConsole.hh"

#include <filesystem>
#include <fstream>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

#include "detail/UniqueFileNamer.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Open a file if it does not already exist
std::ofstream open_file_noreplace(std::string const& filename)
{
    std::ofstream ofs;
    auto flags = std::ios::out;
#if __cplusplus >= 202302L
    // C++23: use the noreplace flag to fail if the file exists
    flags |= std::ios::noreplace;
#else
    // C++17/C++20: check non-atomically if the file exists
    if (std::filesystem::exists(filename))
    {
        // Do not open
        return ofs;
    }
#endif
    ofs.open(filename, flags);
    return ofs;
}

//---------------------------------------------------------------------------//
}  // namespace

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
 * Construct with filename and open mode.
 */
FileOrStdout::FileOrStdout(std::string filename, OpenMode mode)
    : filename_{std::move(filename)}
{
    CELER_VALIDATE(!filename_.empty(),
                   << "empty filename is not valid for output");
    if (filename_ == "-")
    {
        filename_ = "<stdout>";
        return;
    }

    // Open the specified file based on the OpenMode
    if (mode == OpenMode::overwrite)
    {
        // Clobber if it exists
        outf_.open(filename_);
    }
    else
    {
        // Try to open without replacing
        outf_ = open_file_noreplace(filename_);
        if (mode == OpenMode::unique && !outf_.is_open() && !outf_.bad())
        {
            auto msg = CELER_LOG(warning);
            msg << "Failed to open file '" << filename_
                << "' without clobbering";

            // Try with a unique filename
            detail::UniqueFileNamer make_filename(filename_);

            int max_trials_{10};
            while (!outf_.is_open() && max_trials_-- > 0)
            {
                filename_ = make_filename();
                outf_ = open_file_noreplace(filename_);
            }

            if (outf_.is_open())
            {
                msg << ": renamed to " << filename_;
            }
        }
    }
    CELER_VALIDATE(outf_.is_open(), << "failed to open '" << filename_ << "'");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
